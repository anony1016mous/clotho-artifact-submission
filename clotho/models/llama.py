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

base_model_name = "meta-llama/Llama-3.1-8B-Instruct"

load_dotenv(dotenv_path='/root/workspace/.env')
huggingface_token = os.getenv("HF_TOKEN")
if huggingface_token is None:
    raise ValueError("HF_TOKEN environment variable is not set.")
login(token=huggingface_token)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token":"<pad>"})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

if device != "cuda":
    print("Warning: CUDA is not available.")
    exit(1)

# Messages Template Example
# messages = [
#     {"role": "system", "content": f"""You are a helpful assistant who can answer the given question. Examples of questions and answers are given below:
# {few_shot_examples}

# You must answer in the same format as the examples above."""},
#     {"role": "user", "content": f"""{input_text}"""}
# ]

def get_hidden_states(input_text, messages_template, produce_inference_results=False, num_inference_runs=NUM_INFERENCE_RUNS):
    # TODO: batch processing (to speed up inference time)
    model.eval()
    with torch.no_grad():
        messages = messages_template(input_text)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs.to(device)

        outputs = model(**inputs, output_hidden_states=True)
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
            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                generated_text = generated_text.replace('<|eot_id|>', '').strip()
            else:
                print(f"Warning: Generated text does not contain assistant header.")
                print(f"*** Generated text: {generated_text}")

            inference_results.append(generated_text)
        
        return {
            "hidden_states": outputs.hidden_states,
            "input_text": input_text,
            "full_prompt": messages,
            "inference_results": inference_results
        }

def check_all_checkpoints_recorded(tokenIndex2states, template_keywords):
    all_checkpoints_recorded = True
    not_checked_checkpoints = []
    
    for checkpoint in template_keywords:
        if checkpoint not in tokenIndex2states:
            all_checkpoints_recorded = False
            not_checked_checkpoints.append(checkpoint)
        
    return all_checkpoints_recorded, not_checked_checkpoints

def get_hidden_states_during_generation(input_text, messages_template, repeat=NUM_INFERENCE_RUNS, target_layers=[16], temperature=0.8, last_token_index_to_record=10, token_record_interval=5, template_keywords=None):
    model.eval()
    with torch.no_grad():        
        messages = messages_template(input_text)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs.to(device)

        inference_results = []
        token_generation_results = []
        hidden_state_snapshots = []
        
        if template_keywords is None or not isinstance(template_keywords, list):
            template_keywords = []
            
        checkpoint2normString = {}
        for checkpoint in template_keywords:
            norm_str = checkpoint.replace(" ", "")
            checkpoint2normString[checkpoint] = norm_str

        for _ in range(repeat):
            curr_input_ids = inputs.input_ids
            generated_tokens = []
            tokenIndex2states = {}
            
            token_index = 0
            while True:
                outputs = model(input_ids=curr_input_ids, output_hidden_states=True)
                next_token_logits = outputs.logits[:, -1, :]
                
                probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_tokens.append(next_token.item())
                
                if (token_index % token_record_interval == 0) and (token_index <= last_token_index_to_record):
                    hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                
                    tokenIndex2states[token_index] = hidden_states

                all_recorded, not_recorded_checkpoints = check_all_checkpoints_recorded(tokenIndex2states, template_keywords)
                if not all_recorded:
                    generated_str = "".join(tokenizer.convert_ids_to_tokens(generated_tokens))
                    generated_str_norm = generated_str.replace("▁", "").replace(" ", "").replace("\u0120", "")
                    
                    for checkpoint in not_recorded_checkpoints:
                        norm_checkpoint_str = checkpoint2normString[checkpoint]
                        if norm_checkpoint_str in generated_str_norm:
                            hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]

                            if checkpoint in tokenIndex2states:
                                print(f"Warning: Checkpoint '{checkpoint}' hidden states already recorded during the generation. Overwriting it.")
                            tokenIndex2states[checkpoint] = hidden_states
                            break
                        
                if next_token.item() == tokenizer.eos_token_id or token_index == MAX_NEW_TOKENS - 1:
                    hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                
                    tokenIndex2states['IE'] = hidden_states # inference end (or premature end due to the output length limit)
                    
                    if token_index == MAX_NEW_TOKENS - 1:
                        print(f"[Warning] Output generation reached the maximum token limit of {MAX_NEW_TOKENS}.")
                        print(f"Input text: {input_text}")
                    
                    break
                
                curr_input_ids = torch.cat([curr_input_ids, next_token.to(curr_input_ids.device)], dim=-1)
                token_index += 1

                del outputs, next_token_logits, probs, next_token
                
            generated_ids = torch.cat([inputs.input_ids, torch.tensor([generated_tokens]).to(device)], dim=-1)
            generated_ids_output_only = torch.tensor(generated_tokens).to(device)
            token_wise_generations = tokenizer.convert_ids_to_tokens(generated_ids_output_only.detach().cpu().clone().tolist(), skip_special_tokens=False)
            generated_text = tokenizer.decode(generated_ids[0].detach().cpu().clone(), skip_special_tokens=False)
            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                generated_text = generated_text.replace('<|eot_id|>', '').strip()
            else:
                print(f"Warning: Generated text does not contain assistant header.")
                print(f"*** Generated text: {generated_text}")
            
            token_generation_results.append(token_wise_generations)
            inference_results.append(generated_text)
            hidden_state_snapshots.append(tokenIndex2states)
            
            del generated_ids, generated_ids_output_only
            del token_wise_generations, generated_text
            
            del curr_input_ids
            del generated_tokens, tokenIndex2states
            
            gc.collect()
            torch.cuda.empty_cache()
            
        del inputs
        
        gc.collect()
        torch.cuda.empty_cache()
            
        return {
            "input_text": input_text,
            "hidden_state_snapshots": hidden_state_snapshots,
            "generated_tokens": token_generation_results,
            "full_prompt": messages,
            "inference_results": inference_results
        }

def get_hidden_states_from_template_keywords(input_text, messages_template, repeat=NUM_INFERENCE_RUNS, target_layers=[16], temperature=0.8, template_keywords=None):
    model.eval()
    with torch.no_grad():        
        messages = messages_template(input_text)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs.to(device)

        inference_results = []
        token_generation_results = []
        hidden_state_snapshots = []
        
        if template_keywords is None or not isinstance(template_keywords, list):
            raise Exception("No template keywords provided. Please provide a list of template keywords to extract hidden states")
            
        checkpoint2normString = {}
        for checkpoint in template_keywords:
            norm_str = checkpoint.replace(" ", "")
            checkpoint2normString[checkpoint] = norm_str

        for _ in range(repeat):
            curr_input_ids = inputs.input_ids
            generated_tokens = []
            tokenIndex2states = {}
            
            token_index = 0
            while True:
                outputs = model(input_ids=curr_input_ids, output_hidden_states=True)
                next_token_logits = outputs.logits[:, -1, :]
                
                probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_tokens.append(next_token.item())
                
                all_recorded, not_recorded_checkpoints = check_all_checkpoints_recorded(tokenIndex2states, template_keywords)
                if not all_recorded:
                    generated_str = "".join(tokenizer.convert_ids_to_tokens(generated_tokens))
                    generated_str_norm = generated_str.replace("▁", "").replace(" ", "").replace("\u0120", "")
                    
                    for checkpoint in not_recorded_checkpoints:
                        norm_checkpoint_str = checkpoint2normString[checkpoint]
                        if norm_checkpoint_str in generated_str_norm:
                            hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]

                            if checkpoint in tokenIndex2states:
                                print(f"Warning: Checkpoint '{checkpoint}' hidden states already recorded during the generation. Overwriting it.")
                            tokenIndex2states[checkpoint] = hidden_states
                            break
                        
                else:
                    break
                
                curr_input_ids = torch.cat([curr_input_ids, next_token.to(curr_input_ids.device)], dim=-1)
                token_index += 1

                del outputs, next_token_logits, probs, next_token
                
            hidden_state_snapshots.append(tokenIndex2states)
            
            del curr_input_ids
            del generated_tokens, tokenIndex2states
            
            gc.collect()
            torch.cuda.empty_cache()
            
        del inputs
        
        gc.collect()
        torch.cuda.empty_cache()
            
        return {
            "input_text": input_text,
            "hidden_state_snapshots": hidden_state_snapshots,
            "full_prompt": messages,
        }

def get_hidden_states_till_n_tokens(input_text, messages_template, n=1, repeat=NUM_INFERENCE_RUNS, target_layer=16, temperature=0.8):
    model.eval()
    with torch.no_grad():        
        messages = messages_template(input_text)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs.to(device)

        outputs = model(**inputs, output_hidden_states=True)

        inference_results = []
        intermediate_states = []

        for _ in range(repeat):
            curr_input_ids = inputs.input_ids
            generated_tokens = []
            tokenIndex2states = []
            
            for i in range(n):
                outputs = model(input_ids=curr_input_ids.to(device), output_hidden_states=True)
                next_token_logits = outputs.logits[:, -1, :]
                
                probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_tokens.append(next_token.item())
                
                hidden_states = outputs.hidden_states[target_layer][:, -1, :].detach().cpu().clone()
                tokenIndex2states.append(hidden_states)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                curr_input_ids = torch.cat([curr_input_ids, next_token.to(curr_input_ids.device)], dim=-1)
                
            generated_ids = torch.cat([inputs.input_ids, torch.tensor([generated_tokens]).to(device)], dim=-1)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                generated_text = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                generated_text = generated_text.replace('<|eot_id|>', '').strip()
            else:
                print(f"Warning: Generated text does not contain assistant header.")
                print(f"*** Generated text: {generated_text}")
            
            inference_results.append(generated_text)
            intermediate_states.append(tokenIndex2states)
            
        return {
            "hidden_states_n_generations": intermediate_states,
            
            "input_text": input_text,
            "full_prompt": messages,
            "partial_inference_results": inference_results
        }


def prepare_inputs_for_token_prob_calculation(prompt, generated_output=None):
    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    
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
