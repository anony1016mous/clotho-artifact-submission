# Calculating perplexity w/ Transformers: https://huggingface.co/docs/transformers/perplexity
# No need for sliding window, as Llama 3.1 models have large context windows

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import login


def calc_confidence_scores(model_wrapper, prompt, generated_output):
    full_input_ids, generated_tokens, prompt_len = model_wrapper.prepare_inputs_for_token_prob_calculation(prompt, generated_output)

    model = model_wrapper.model
    model.eval()
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits # (batch_size, sequence_length, vocab_size)
        
        shift_logits = logits[:, :-1, :].contiguous() # (batch_size, sequence_length - 1, vocab_size)
        shift_input_ids = full_input_ids[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_input_ids.unsqueeze(-1)).squeeze(-1)
        
        output_log_probs = token_log_probs[:, prompt_len - 1:]
        avg_log_prob = output_log_probs.mean()
        
        perplexity = torch.exp(-avg_log_prob).item()
        avg_log_prob = avg_log_prob.item()
        
        return {
            "output_log_probs": output_log_probs.squeeze(0).to(torch.float16).cpu().numpy(),
            "average_log_probs": avg_log_prob,
            "perplexity": perplexity,
            "generated_tokens": generated_tokens,
        }

def calc_average_entropy(model_wrapper, prompt, generated_output, include_eos= False):
    full_input_ids, generated_tokens, prompt_len = model_wrapper.prepare_inputs_for_token_prob_calculation(prompt, generated_output)
    
    model = model_wrapper.model
    model.eval()
    
    device = next(model.parameters()).device
    full_input_ids = full_input_ids.to(device)
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits # [1, seq_len, vocab]
        
        shift_logits = logits[:, :-1, :] # [1, seq_len-1, vocab]
        shift_input_ids = full_input_ids[:, 1:] # [1, seq_len-1]

        log_probs = F.log_softmax(shift_logits, dim=-1)  # [1, L-1, V]
        probs = log_probs.exp()

        start = max(0, prompt_len - 1)
        end = shift_logits.shape[1]

        if not include_eos:
            end = max(start, end - 1)

        if end <= start:
            return 0.0

        # Entropy per step: H_t = -∑ p log p
        slice_log_probs = log_probs[:, start:end, :]
        slice_probs = probs[:, start:end, :]
        per_token_ent = -(slice_probs * slice_log_probs).sum(dim=-1)
        avg_entropy = per_token_ent.mean().item()
    
    return {
        "per_token_entropies": per_token_ent.to(torch.float32).cpu().numpy(),
        "average_entropy": float(avg_entropy.item() if hasattr(avg_entropy, "item") else avg_entropy),
        "generated_tokens": generated_tokens,
    }

def _calc_confidence_scores(model_wrapper, prompt, generated_output):
    device = model_wrapper.device
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    
    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    full_text = prompt + generated_output
    
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    full_input_ids = tokenizer.encode(full_text, return_tensors='pt', add_special_tokens=False).to(device)
    
    eos_token_id = tokenizer.eos_token_id
    eos_tensor = torch.tensor([[eos_token_id]], device=device)
    full_input_ids = torch.cat([full_input_ids, eos_tensor], dim=1)
    
    prompt_len = prompt_ids.shape[1]
    full_len = full_input_ids.shape[1]
    
    tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0])
    generated_tokens = tokens[prompt_len:]
    
    model.eval()
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits # (batch_size, sequence_length, vocab_size)
        
        shift_logits = logits[:, :-1, :].contiguous() # (batch_size, sequence_length - 1, vocab_size)
        shift_input_ids = full_input_ids[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_input_ids.unsqueeze(-1)).squeeze(-1)
        
        output_log_probs = token_log_probs[:, prompt_len - 1:]
        avg_log_prob = output_log_probs.mean()
        
        perplexity = torch.exp(-avg_log_prob).item()
        avg_log_prob = avg_log_prob.item()
        
        return {
            "output_log_probs": output_log_probs.squeeze(0).to(torch.float16).cpu().numpy(),
            "average_log_probs": avg_log_prob,
            "perplexity": perplexity,
            "generated_tokens": generated_tokens,
        }

def calc_confidence_scores_input(model_wrapper, prompt):
    full_input_ids, generated_tokens, _ = model_wrapper.prepare_inputs_for_token_prob_calculation(prompt)
    model = model_wrapper.model
    model.eval()
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits # (batch_size, sequence_length, vocab_size)

        shift_logits = logits[:, :-1, :].contiguous() # (batch_size, sequence_length - 1, vocab_size)
        shift_input_ids = full_input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_input_ids.unsqueeze(-1)).squeeze(-1)

        avg_log_prob = token_log_probs.mean()
        perplexity = torch.exp(-avg_log_prob).item()
        avg_log_prob = avg_log_prob.item()

        return {
            "input_log_probs": token_log_probs.squeeze(0).to(torch.float16).cpu().numpy(),
            "average_log_probs": avg_log_prob,
            "perplexity": perplexity,
            "generated_tokens": generated_tokens,
        }

if __name__ == "__main__":
    import clotho.models.llama as llama_model_wrapper
    import clotho.dataset.spell_check.templates as templates_spell_check
    
    input_text = "motives inspired by Mammon were often inextricably blended with tings pertaining to Caesar and to God"
    inference = "Reasoning: The word \"tings\" does not seem to match common English words.\nMisspelled word: tings\nCorrected word: things"
    
    messages = templates_spell_check.messages_template(input_text)

    res = calc_confidence_scores(
        model_wrapper=llama_model_wrapper,
        prompt=messages,
        generated_output=inference
    )
    
    output_log_probs = res["output_log_probs"]
    average_log_probs = res["average_log_probs"]
    perplexity = res["perplexity"]

    print("Input text: ", input_text)    
    print("Output log probabilities: ", output_log_probs)
    print("Average log probs (among generated tokens):", average_log_probs)
    print("Perplexity:", perplexity)
