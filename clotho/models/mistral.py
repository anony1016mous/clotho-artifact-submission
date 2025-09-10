import os, gc, json
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 512
NUM_INFERENCE_RUNS = 10

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"

load_dotenv(dotenv_path='/root/workspace/.env')
huggingface_token = os.getenv("HF_TOKEN")
if not huggingface_token:
    raise ValueError("HF_TOKEN environment variable is not set.")
login(token=huggingface_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print("Warning: CUDA is not available.")
    exit(1)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    token=huggingface_token,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, token= huggingface_token, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
model.config.pad_token_id = tokenizer.pad_token_id

def _decode_clean_all(input_ids, full_ids, tokenizer):
    return tokenizer.decode(full_ids[0], skip_special_tokens=True).strip()

def _decode_clean(input_ids, full_ids, tokenizer):
    gen_part = full_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_part, skip_special_tokens=True).strip()

def _normalize_text_for_checkpoint_match(s):
    return (
        s.replace("▁", "")
         .replace("Ġ", "")
         .replace(" ", "")
         .replace("\u0120", "")
    )

def check_all_checkpoints_recorded(tokenIndex2states, template_keywords):
    all_ok = True
    not_checked = []
    for cp in template_keywords:
        if cp not in tokenIndex2states:
            all_ok = False
            not_checked.append(cp)
    return all_ok, not_checked

@torch.no_grad()
def get_hidden_states_during_generation_(
    input_text,
    messages_template,
    repeat=NUM_INFERENCE_RUNS,
    target_layers=[16, 22],
    temperature=0.8,
    last_token_index_to_record=-1,
    token_record_interval=5,
    template_keywords=None,
):
    model.eval()
    messages = messages_template(input_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    inference_results = []
    token_generation_results = []
    hidden_state_snapshots = []
    
    if template_keywords is None or not isinstance(template_keywords, list):
        template_keywords = []
    
    checkpoint2normString = {cp: _normalize_text_for_checkpoint_match(cp) for cp in template_keywords}

    for _ in range(repeat):
        curr_input_ids = inputs["input_ids"]
        generated_tokens = []
        tokenIndex2states = {}
        
        token_index = 0
        while True:
            outputs = model(input_ids=curr_input_ids, output_hidden_states=True)
            next_token_logits = outputs.logits[:, -1, :]
            
            probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens.append(next_token.item())

            # fixed-interval snapshots
            if (token_index % token_record_interval == 0) and (token_index <= last_token_index_to_record):
                hs = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                tokenIndex2states[token_index] = hs

            # keyword snapshots
            all_recorded, not_recorded_checkpoints = check_all_checkpoints_recorded(tokenIndex2states, template_keywords)
            if not all_recorded:
                generated_str = "".join(tokenizer.convert_ids_to_tokens(generated_tokens))
                generated_str_norm = _normalize_text_for_checkpoint_match(generated_str)

                for checkpoint in not_recorded_checkpoints:
                    norm_checkpoint_str = checkpoint2normString[checkpoint]
                    if norm_checkpoint_str in generated_str_norm:
                        hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                        if checkpoint in tokenIndex2states:
                            print(f"Warning: Checkpoint '{checkpoint}' hidden states already recorded during the generation. Overwriting it.")
                        tokenIndex2states[checkpoint] = hidden_states
                        break

            if (next_token.item() == tokenizer.eos_token_id) or (token_index == MAX_NEW_TOKENS - 1):
                hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                tokenIndex2states['IE'] = hidden_states
                if token_index == MAX_NEW_TOKENS - 1:
                    print(f"[Warning] Output generation reached the maximum token limit of {MAX_NEW_TOKENS}.")
                    print(f"Input text: {input_text}")
                break

            curr_input_ids = torch.cat([curr_input_ids, next_token.to(curr_input_ids.device)], dim=-1)
            token_index += 1

            del outputs, next_token_logits, probs, next_token

        generated_ids = torch.cat([inputs["input_ids"], torch.tensor([generated_tokens]).to(model.device)], dim=-1)
        generated_ids_output_only = torch.tensor(generated_tokens, device=model.device)

        token_wise_generations = tokenizer.convert_ids_to_tokens(
            generated_ids_output_only.detach().cpu().tolist(),
            skip_special_tokens=False
        )
        generated_text = _decode_clean(inputs["input_ids"], generated_ids, tokenizer)

        token_generation_results.append(token_wise_generations)
        inference_results.append(generated_text)
        hidden_state_snapshots.append(tokenIndex2states)

        del generated_ids, generated_ids_output_only, token_wise_generations, generated_text, curr_input_ids, generated_tokens, tokenIndex2states
        gc.collect(); torch.cuda.empty_cache()

    del inputs
    gc.collect(); torch.cuda.empty_cache()

    return {
        "input_text": input_text,
        "hidden_state_snapshots": hidden_state_snapshots,
        "generated_tokens": token_generation_results,
        "full_prompt": messages,
        "inference_results": inference_results
    }

@torch.no_grad()
def get_hidden_states_during_generation(input_text, messages_template, repeat=NUM_INFERENCE_RUNS, target_layers=[16, 22], temperature=0.8, last_token_index_to_record=-1, token_record_interval=5, template_keywords=None):
    """Cacheing hidden states during generation."""
    model.eval()
    messages = messages_template(input_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids, device=input_ids.device))
    
    inference_results = []
    token_generation_results = []
    hidden_state_snapshots = []

    if not isinstance(template_keywords, list):
        template_keywords = []
    checkpoint2norm = {cp: _normalize_text_for_checkpoint_match(cp) for cp in template_keywords}
    
    for _ in range(repeat):
        past_key_values = None
        curr_ids = input_ids
        curr_mask = attention_mask
        generated_tokens = []
        tokenIndex2states = {}
        
        token_index = 0
        while True:
            outputs = model(input_ids=curr_ids, attention_mask=curr_mask, use_cache=True, past_key_values=past_key_values, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_id = next_token.item()
            generated_tokens.append(next_id)
            
            # fixed interval snapshots
            if (token_index % token_record_interval == 0) and (token_index <= last_token_index_to_record):
                hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                tokenIndex2states[token_index] = hidden_states
            
            # keyword snapshots
            all_recorded, not_recorded_checkpoints = check_all_checkpoints_recorded(tokenIndex2states, template_keywords)
            if not all_recorded:
                gen_decoded = tokenizer.decode(generated_tokens, skip_special_tokens=False)
                gen_norm = _normalize_text_for_checkpoint_match(gen_decoded)
                for checkpoint in not_recorded_checkpoints:
                    if checkpoint2norm[checkpoint] in gen_norm:
                        hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                        tokenIndex2states[checkpoint] = hidden_states
                        break

            if (next_id == tokenizer.eos_token_id) or (token_index == MAX_NEW_TOKENS - 1):
                hidden_states = [(l, outputs.hidden_states[l][:, -1, :].detach().cpu().clone()) for l in target_layers]
                tokenIndex2states['IE'] = hidden_states
                if token_index == MAX_NEW_TOKENS - 1:
                    print(f"[Warning] Reached MAX_NEW_TOKENS={MAX_NEW_TOKENS}.")
                break
            
            # With KV cache, feed only the last token next step
            curr_ids = next_token.to(input_ids.device)
            curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=curr_mask.device)], dim=-1)

            token_index += 1
            
            del outputs, next_token_logits, probs, next_token

        fully_generated = torch.tensor([generated_tokens], device=input_ids.device)
        generated_ids = torch.cat([input_ids, fully_generated], dim=-1)

        token_wise_generations = tokenizer.convert_ids_to_tokens(fully_generated[0].tolist(), skip_special_tokens=False)
        generated_text = _decode_clean(input_ids, generated_ids, tokenizer)

        token_generation_results.append(token_wise_generations)
        inference_results.append(generated_text)
        hidden_state_snapshots.append(tokenIndex2states)

        del fully_generated, generated_ids, token_wise_generations, generated_text
        gc.collect(); torch.cuda.empty_cache()

    del inputs, input_ids, attention_mask
    gc.collect(); torch.cuda.empty_cache()

    return {
        "input_text": input_text,
        "hidden_state_snapshots": hidden_state_snapshots,
        "generated_tokens": token_generation_results,
        "full_prompt": messages,
        "inference_results": inference_results,
    }

def get_hidden_states(input_text, messages_template, produce_inference_results=False, num_inference_runs=NUM_INFERENCE_RUNS):
    model.eval()
    with torch.no_grad():
        messages = messages_template(input_text)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=False, truncation=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # last_layer_hidden = outputs.hidden_states[-1]  # shape: (num_hidden_layers, batch_size, seq_len, hidden_size)
        # mid_layer_hidden = outputs.hidden_states[len(outputs.hidden_states) // 2]
        # first_layer_hidden = outputs.hidden_states[0]

        if produce_inference_results:
            raise NotImplementedError("Inference result production is not implemented.")

        return {
            "hidden_states": outputs.hidden_states,
        }

def prepare_inputs_for_token_prob_calculation(prompt, generated_output=None):
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

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
