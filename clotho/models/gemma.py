import os
import gc
import json
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 512
NUM_INFERENCE_RUNS = 10

base_model_name = "google/gemma-2-9b-it"

load_dotenv(dotenv_path='/root/workspace/.env')
huggingface_token = os.getenv("HF_TOKEN")
if huggingface_token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")
login(token=huggingface_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print("Warning: CUDA is not available.")
    exit(1)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
    
def _decode_clean_all(generated_ids):
    """Decode generated tokens to text, handling special tokens."""
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if not text.strip():
        # Fallback: decode w/o skipping, then strip all known specials
        raw = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        for sp in tokenizer.all_special_tokens:
            raw = raw.replace(sp, "")
        text = raw.strip()
    return text

def _decode_clean(generated_ids):
    """Decode only the assistant's output, removing prompt/system parts."""
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if not text.strip():
        raw = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        for sp in tokenizer.all_special_tokens:
            raw = raw.replace(sp, "")
        text = raw.strip()

    if "model\n" in text:
        text = text.split("model\n", 1)[1].strip()
    elif "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1].strip()
    return text

def inline_system_as_user(message):
    """
    Convert [system, user, ...] to a Gemma custom prompt by 
    inlining the system content into the first user turn.
    """
    if not message:
        return message
    
    if len(message) >= 2 and message[0]["role"] == "system" and message[1]["role"] == "user":
        system_content = message[0]["content"].strip()
        user_content = message[1]["content"].strip()

        merged_content = f"{system_content}\n\n{user_content}"
        new_message = [{"role": "user", "content": merged_content}]
        new_message.extend(message[2:])
        return new_message

    return message

def check_all_checkpoints_recorded(tokenIndex2states, template_keywords):
    all_checkpoints_recorded = True
    not_checked_checkpoints = []
    for checkpoint in template_keywords:
        if checkpoint not in tokenIndex2states:
            all_checkpoints_recorded = False
            not_checked_checkpoints.append(checkpoint)
    return all_checkpoints_recorded, not_checked_checkpoints

def get_hidden_states_during_generation(input_text, messages_template, repeat=NUM_INFERENCE_RUNS, target_layers=[21, 28], temperature=0.8, last_token_index_to_record=-1, token_record_interval=5, template_keywords=None):
    model.eval()
    with torch.inference_mode():
        messages = messages_template(input_text)
        messages_formatted = inline_system_as_user(messages)
        prompt = tokenizer.apply_chat_template(messages_formatted, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        inference_results = [] # final decoded strings
        token_generation_results = [] # list[list[str]]: token-wise generations
        hidden_state_snapshots = [] # list[dict]: token index / checkpoint -> [(layer, tensor), ...]

        if template_keywords is None or not isinstance(template_keywords, list):
            template_keywords = []
        checkpoint2normString = {cp: cp.replace(" ", "") for cp in template_keywords}

        # Layer index sanity
        num_layers = model.config.num_hidden_layers  # Gemma2-9B: 42
        target_layers = [l for l in target_layers if 0 <= l <= num_layers]
        if not target_layers:
            raise ValueError(f"target_layers empty after filter; valid range is 0..{num_layers} (0=embeddings, 1..N=post-block).")

        for _ in range(repeat):
            curr_input_ids = inputs.input_ids
            generated_tokens = []
            tokenIndex2states = {}

            token_index = 0
            while True:
                outputs = model(input_ids=curr_input_ids, output_hidden_states=True, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]

                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                next_token_id = int(next_token.item())
                generated_tokens.append(next_token_id)

                # fixed interval snapshots
                if (token_index % token_record_interval == 0) and (token_index <= last_token_index_to_record):
                    hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                    tokenIndex2states[token_index] = hidden_states

                # keyword snapshots
                all_recorded, not_recorded_checkpoints = check_all_checkpoints_recorded(tokenIndex2states, template_keywords)
                if not all_recorded:
                    gen_tokens_text = tokenizer.convert_ids_to_tokens(generated_tokens)
                    generated_str = "".join(gen_tokens_text)
                    generated_str_norm = generated_str.replace("▁", "").replace(" ", "").replace("\u0120", "")
                    for checkpoint in not_recorded_checkpoints:
                        norm_cp = checkpoint2normString[checkpoint]
                        if norm_cp in generated_str_norm:
                            hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                            if checkpoint in tokenIndex2states:
                                print(f"Warning: checkpoint '{checkpoint}' already recorded; overwriting.")
                            tokenIndex2states[checkpoint] = hidden_states
                            break
                
                EOT_ID = tokenizer.convert_tokens_to_ids("<end_of_turn>")
                hit_eot = (EOT_ID is not None) and (next_token_id == EOT_ID)
                hit_max = (token_index == MAX_NEW_TOKENS - 1)

                if hit_eot or hit_max:
                    hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                    tokenIndex2states[f"IE"] = hidden_states
                    if hit_max and not hit_eot:
                        print(f"[Warning] Reached MAX_NEW_TOKENS={MAX_NEW_TOKENS} without EOT.")
                        print(f"Input text: {input_text}")
                    break

                curr_input_ids = torch.cat([curr_input_ids, next_token.to(curr_input_ids.device)], dim=-1)
                token_index += 1

                del outputs, next_token_logits, probs, next_token

            token_wise_generations = tokenizer.convert_ids_to_tokens(torch.tensor(generated_tokens).tolist(), skip_special_tokens=False)
            generated_ids = torch.cat([inputs.input_ids, torch.tensor([generated_tokens]).to(device)], dim=-1)
            generated_text = _decode_clean(generated_ids)

            token_generation_results.append(token_wise_generations)
            inference_results.append(generated_text)
            hidden_state_snapshots.append(tokenIndex2states)

            del generated_ids, token_wise_generations, generated_text
            del curr_input_ids, generated_tokens, tokenIndex2states
            gc.collect(); torch.cuda.empty_cache()

        del inputs
        gc.collect(); torch.cuda.empty_cache()

        return {
            "input_text": input_text,
            "hidden_state_snapshots": hidden_state_snapshots,
            "generated_tokens": token_generation_results,
            "full_prompt": messages_formatted,
            "inference_results": inference_results
        }
        
def get_hidden_states(input_text, messages_template, produce_inference_results=False, num_inference_runs=NUM_INFERENCE_RUNS):
    model.eval()
    with torch.inference_mode():
        messages = messages_template(input_text)
        messages_formatted = inline_system_as_user(messages)
        prompt = tokenizer.apply_chat_template(messages_formatted, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # last_layer_hidden = outputs.hidden_states[-1]  # shape: (num_hidden_layers, batch_size, seq_len, hidden_size)
        # mid_layer_hidden = outputs.hidden_states[len(outputs.hidden_states) // 2]
        # first_layer_hidden = outputs.hidden_states[0]
        
        if not produce_inference_results:
            return {
                "hidden_states": outputs.hidden_states,
            }

        inference_results = []

        for _ in range(num_inference_runs):
            generated_ids = model.generate(**inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            if "<start_of_turn>model" in generated_text:
                generated_text = generated_text.split("<start_of_turn>model")[-1]
                generated_text = generated_text.replace('<end_of_turn><eos>', '').strip()
            else:
                print(f"Warning: Generated text does not contain assistant header.")
                print(f"*** Generated text: {generated_text}")

            inference_results.append(generated_text)

        return {
            "hidden_states": outputs.hidden_states,
            "inference_results": inference_results,
            "full_prompt": messages,
            "input_text": input_text
        }


def prepare_inputs_for_token_prob_calculation(prompt, generated_output=None):
    messages_formatted = inline_system_as_user(prompt)
    prompt = tokenizer.apply_chat_template(messages_formatted, tokenize=False, add_generation_prompt=True)

    if generated_output is None:
        full_text = prompt
    else:
        full_text = prompt + generated_output
    
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    full_input_ids = tokenizer.encode(full_text, return_tensors='pt', add_special_tokens=False).to(device)
    
    if generated_output is not None:
        eos_token_id = tokenizer.eos_token_id
        eos_tensor = torch.tensor([[eos_token_id]], device=device)
        full_input_ids = torch.cat([full_input_ids, eos_tensor], dim=1)
    
    prompt_len = prompt_ids.shape[1]
    full_len = full_input_ids.shape[1]
    
    tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0])
    if generated_output is None:
        tokens = tokens
    else:
        tokens = tokens[prompt_len:]
    
    return full_input_ids, tokens, prompt_len
