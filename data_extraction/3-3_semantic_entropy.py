import os, sys

import argparse

from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from huggingface_hub import login
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from clotho.exp_config import task2template, task2input_key
from clotho.metrics.semantic_entropy import pipeline, EntailmentDeberta

if __name__ == "__main__":
    load_dotenv(dotenv_path='/root/workspace/.env')
    huggingface_token = os.getenv("HF_TOKEN")
    if not huggingface_token:
        raise ValueError("HF_TOKEN environment variable is not set.")
    login(token=huggingface_token)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    parser = argparse.ArgumentParser(description="Compute Semantic Uncertainty for generated responses.")
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        help= "Model to use for inference (ex: llama)",
    )
    parser.add_argument(
        "--dataset_type",
        "-t",
        type=str,
        help="Type of dataset to use (ex: spell_check)."
    )
    parser.add_argument(
        "--dataset_name",
        "-d",
        type=str,
        help="Name of the dataset to use (ex: misspell_injected_wordnet)."
    )
    parser.add_argument(
        "--prompt_template",
        "-p",
        type=str,
        help="Name of the prompt template to use (ex: messages_template)."
    )
    args = parser.parse_args()
    
    MODEL_NAME = args.model_name
    DATASET_TYPE = args.dataset_type
    DATASET_NAME = args.dataset_name # expected to be in jsonl format
    prompt_template_name = args.prompt_template
    
    if MODEL_NAME == 'llama':
        import clotho.models.llama as model_wrapper
    elif MODEL_NAME == 'gemma':
        import clotho.models.gemma as model_wrapper
    elif MODEL_NAME == 'mistral':
        import clotho.models.mistral as model_wrapper
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}. Available models: {['llama', 'gemma', 'mistral']}")

    prompt_template_func = task2template[DATASET_TYPE][prompt_template_name]
    
    json_input_file = f"./results_{MODEL_NAME}/{DATASET_TYPE}/inference_results/{prompt_template_name}/{DATASET_NAME}_R10_T0.8.json"
    json_output_file = f"./metrics/results/semantic_entropy/results_{MODEL_NAME}/{DATASET_TYPE}_{DATASET_NAME}_R10_T0.8.json"

    print("Loading DeBERTa model...")
    deberta_model = EntailmentDeberta()

    print("Processing JSON file...")
    results = pipeline(
        dataset_name=DATASET_NAME,
        dataset_type=DATASET_TYPE,
        task2input_key=task2input_key,
        input_filepath=json_input_file,
        output_filepath=json_output_file,
        entailment_model=deberta_model,
        model_wrapper=model_wrapper,
        templates=prompt_template_func
    )
    print("Processing complete!")