import os, sys
import logging
import json

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .logprobs import calc_confidence_scores

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class BaseEntailment:
    """Base class for entailment models."""
    def check_implication(self, text1, text2, **kwargs):
        """
        Returns: 0: contradiction / 1: neutral  / 2: entailment
        """
        raise NotImplementedError
    
    def save_prediction_cache(self, filepath = None):
        """Save prediction cache to file."""
        pass

class EntailmentDeberta(BaseEntailment):
    """DeBERTa-based entailment model."""
    def __init__(self):
        logger.info("Loading DeBERTa entailment model...")
        entailment_model= "microsoft/deberta-v2-xlarge-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(entailment_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(entailment_model).to(DEVICE)
        logger.info("DeBERTa model loaded successfully")

    def check_implication(self, text1, text2, **kwargs):
        """Check if text1 implies text2 using DeBERTa."""
        inputs = self.tokenizer(text1, text2, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(F.softmax(logits, dim=1)).cpu().item()
        return prediction

def get_semantic_ids(strings_list, model, strict_entailment = True, **kwargs):
    """Group predictions into semantic clusters."""
    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2, **kwargs)
        implication_2 = model.check_implication(text2, text1, **kwargs)
        
        if strict_entailment:
            # Both directions must be entailment
            return (implication_1 == 2) and (implication_2 == 2)
        else:
            # Neither should be contradiction, and not both neutral
            implications = [implication_1, implication_2]
            return (0 not in implications) and (implications != [1, 1])
    
    semantic_ids = [-1] * len(strings_list)
    next_id = 0
    
    for i, string1 in enumerate(strings_list):
        if semantic_ids[i] == -1:  # Not yet assigned
            semantic_ids[i] = next_id
            
            for j in range(i + 1, len(strings_list)):
                if semantic_ids[j] == -1 and are_equivalent(string1, strings_list[j]):
                    semantic_ids[j] = next_id
                    
            next_id += 1
    return semantic_ids

def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id."""
    unique_ids = sorted(set(semantic_ids))
    log_likelihood_per_semantic_id = []
    
    for uid in unique_ids:
        # Find all positions with this semantic ID
        id_indices = [i for i, sid in enumerate(semantic_ids) if sid == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        
        if agg == 'sum_normalized':
            log_normalizer = np.log(np.sum(np.exp(log_likelihoods)))
            log_lik_norm = np.array(id_log_likelihoods) - log_normalizer
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")        
        log_likelihood_per_semantic_id.append(logsumexp_value)
    return log_likelihood_per_semantic_id

def predictive_entropy(log_probs):
    """Compute predictive entropy from log probabilities.
    
    Standard entropy: E[-log p(x)] = -1/N sum_i log p(x_i)
    """
    return -np.sum(log_probs) / len(log_probs)

def predictive_entropy_rao(log_probs):
    """Compute Rao entropy from log probabilities.
    
    Rao entropy: -sum_i p_i log p_i
    """
    probs = np.exp(log_probs)
    return -np.sum(probs * log_probs)

def compute_semantic_uncertainty(entailment_model, responses, log_likelihoods, strict_entailment=True, question=None):
    """Compute various uncertainty measures for a set of responses."""
    results = {}
    
    kwargs = {'question': question} if question else {}
    semantic_ids = get_semantic_ids(responses, entailment_model, strict_entailment, **kwargs)
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_likelihoods)
    results['semantic_entropy'] = predictive_entropy_rao(log_likelihood_per_semantic_id)
    results['num_semantic_clusters'] = len(set(semantic_ids))
    results['num_responses'] = len(responses)
    return results, semantic_ids

def pipeline(dataset_name, dataset_type, task2input_key, input_filepath, output_filepath, entailment_model, model_wrapper, templates):
    """Process all entries in the JSON file and compute semantic entropy for each."""
    DATASET_TYPE = dataset_type
    DATASET_NAME = dataset_name

    dataset = []
    with open(f'/root/workspace/clotho/dataset/{DATASET_TYPE}/{DATASET_NAME}.jsonl') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)

    input_texts = []
    for input_index, input_text in enumerate(dataset):
        if task2input_key[DATASET_TYPE] is None:
            input_variables = input_text
        else:
            input_variables = input_text[task2input_key[DATASET_TYPE]]
        input_texts.append(input_variables)
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    results = []
    for idx, entry in enumerate(tqdm(result_data)):
        logger.info(f"Processing entry {idx + 1}/{len(result_data)}")
        inferences = entry.get("inferences", [])
        messages = templates(input_texts[idx])

        log_likelihoods = []
        for inference in inferences:
            try:
                res = calc_confidence_scores(
                    model_wrapper=model_wrapper, 
                    prompt=messages, 
                    generated_output=inference
                )
                log_likelihood = res["average_log_probs"]
                log_likelihoods.append(log_likelihood)
            except Exception as e:
                logger.error(f"Error calculating log likelihood for entry {idx}: {e}")
                log_likelihoods.append(-10.0)  # Default low probability
        try:
            uncertainties, semantic_ids = compute_semantic_uncertainty(
                entailment_model=entailment_model,
                responses=inferences,
                log_likelihoods=log_likelihoods,
            )
            
            # Store results
            entry_result = {
                'entry_index': idx,
                'inferences': inferences,
                'log_likelihoods': log_likelihoods,
                'semantic_ids': semantic_ids,
                'uncertainties': uncertainties
            }
            results.append(entry_result)
            logger.info(f"Entry {idx}: {uncertainties['num_semantic_clusters']} clusters, semantic_entropy: {uncertainties['semantic_entropy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error computing semantic uncertainty for entry {idx}: {e}")
            continue
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if results:
        semantic_entropies = [r['uncertainties']['semantic_entropy'] for r in results]
        num_clusters = [r['uncertainties']['num_semantic_clusters'] for r in results]
        
        print(f"\nProcessed {len(results)} entries successfully")
        print(f"Semantic entropy - Mean: {np.mean(semantic_entropies):.4f}, "
              f"Std: {np.std(semantic_entropies):.4f}")
        print(f"Number of clusters - Mean: {np.mean(num_clusters):.2f}, "
              f"Std: {np.std(num_clusters):.2f}")
    return results