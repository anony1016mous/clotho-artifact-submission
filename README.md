# Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs

This repository contains the code and data for the paper "Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs".

## Reproducing Exploratory and Main Study Analyses

The analysis scripts and results data (Jupyter Notebooks) used in our study are available in the `clotho/RQ` directory.

- **`dataset_statistics.ipynb`**  
  Summarises dataset statistics (e.g., size, pass/fail rates per task and model).

- **`exploratory_study_1_task_specific.ipynb`**  
  Exploratory study (Sec. 2.1): task-specific separability of hidden states.

- **`exploratory_study_2_layer_selection.ipynb`**  
  Exploratory study (Sec. 2.2): identifying informative Transformer layers.

- **`RQ1-1_compare_before_gen_metrics_all_models.ipynb`**  
  RQ1-1: Compare pre-generation adequacy metrics across models (Tab. 3).

- **`RQ1-1_score_visualization.ipynb`**  
  RQ1-1: Visualise predicted scores vs. actual failure rates on a projected 2D plane (Fig. 4).

- **`RQ1-2_compare_sampling_methods.ipynb`**  
  RQ1-2: Compare sampling strategies (Fig. 5, Fig. 6).

- **`RQ2_ROC_AUC.ipynb`**  
  RQ2: Compute ROC-AUC for failure prediction and Mann-Whitney U Test (Tab. 4).

- **`RQ2_test_prioritization.ipynb`**  
  RQ2: Evaluate test prioritisation performance (Tab. 5).

- **`RQ3-1_loh_metrics_evaluation.ipynb`**  
  RQ3-1: Evaluate Clotho vs. post-generation metrics (Tab. 6).

- **`RQ3-2_loh_complementary_metrics.ipynb`**  
  RQ3-2: Analyse complementarity with post-generation uncertainty metrics (Fig. 7).

- **`RQ4_proprietary_model.ipynb`**  
  RQ4: Transfer scores from studied OLMs to GPT, Claude, and Gemini (Tab. 7, Tab. 8, Fig. 8).

- **`supplementary_RQ3_logistic_regression.ipynb`**  
  Supplementary: Additional analysis combining Clotho with a subset of post-generation metrics.

- **`figures/`**  
  Stores visualisations used in the paper.


## Task-specific LLM Prompts Dataset
The prompt templates, input data, and scripts used to synthesize (parts of) the datasets are available in the `clotho/datasets` directory.

For each task, `templates.py` defines the prompt templates with task-specific instructions and output format specifications, `output_parsers.py` implements the logic for parsing model outputs, and the input data files (e.g., `*.jsonl`) provide the test input instances used to generate LLM outputs.

```
.
|-- adding_odd_numbers
|   |-- integer_sequences_length_1_to_10_uniform.jsonl
|   |-- output_parsers.py
|   |-- templates.py
|   `-- test_generator.py    # Script to generate the random integer sequences
|-- github_typo_check
|   |-- github_typo_corpus_cleaned.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- json_repair
|   |-- synthetic_invalid_json    # Scripts to generate synthetic invalid JSON inputs
|   |-- invalid_json_dataset_2166.jsonl
|   |-- invalid_json_dataset_4397.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- model_name_extraction
|   |-- synthetic_abstract    # Scripts to generate synthetic abstracts
|   |-- ml_arxiv_papers_labelling    # Scripts to label ML arXiv papers with model names
|   |-- synthetic_abstracts_gpt4o_3600.jsonl
|   |-- ml_arxiv_papers_no_conflicting_labels.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- pos_detection
|   |-- cleaned_and_sampled_pos_tags.jsonl
|   |-- cleaned_and_sampled_pos_tags_trainset.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- spell_check
|   |-- misspell_injected_wordnet.jsonl
|   |-- output_parsers.py
|   `-- templates.py
|-- syntactic_bug_detection
|   |-- syntactic_bug_injected.jsonl
|   |-- output_parsers.py
|   `-- templates.py
`-- topic_classification
    |-- ag_news_test.jsonl
    |-- output_parsers.py
    `-- templates.py
```


## Download Raw Data

To actually run the notebooks for reproducing our analyses, please download the following raw data files containing the generated LLM outputs and intermediate records of metric scores:

- [raw_results.tar.gz](https://drive.google.com/file/d/1uN_q6eIlkpYc1hJxzVzzqlVc67QGS2jP/view?usp=sharing)
  - Locate the compressed file in `clotho/` directory and extract it there: `results_llama`, `results_gemma`, `results_mistral`, and so on will be created.
- [results_clotho_10seeds.tar.gz](https://drive.google.com/file/d/1cknccBEDcCfMOzNhBwXP87c5I1wZHOGK/view?usp=sharing)
  - Locate the compressed file in `experiments/` directory and extract it there: `results_GMM` will be created.

Due to their large size, we could not include the raw hidden states extracted from LLMs here.
We plan to make them publicly available via Zenodo (or a similar service) upon formal publication of the paper.

## Docker Container Setup
**Step 1. Build Docker Image**
```bash
> cd docker # Dockerfile is in this directory
> docker build -f Dockerfile -t clotho-image .
```

**Step 2. Run Docker Container**

Run this command from the root of the repository (requires NVIDIA Docker runtime for GPU support):

```bash
docker run -dt --gpus all --entrypoint=/bin/bash -v .:/root/workspace --name clotho clotho-image:latest
```

To attach to the running container, use:

```bash
docker exec -it clotho /bin/bash
```

You may have to additionally run
```bash
pip install -r requirements.txt
pip install -e . # install the clotho package with editable mode
```

## Package Structure
The `clotho` toolkit, provided as a Python package, includes code for dataset construction used in our study, hidden state extraction from Hugging Face–deployed open LLMs, and functions for computing both baseline and Clotho’s adequacy metrics based on the extracted hidden state vectors.

The detailed package structure and descriptions are as follows:

```
.
|-- RQ                                    # Jupyter notebooks for exploratory and main study analyses (RQ1–RQ4)
|-- al
|   `-- sampling.py                       # Sampling strategies for reference set construction: random, exploration (diversity), exploitation (uncertainty), balanced
|-- dataset                               # Dataset construction and task-specific prompt preparation
|-- inference.py                          # Inference pipeline for HuggingFace–deployed open LLMs
|-- metrics
|   |-- reference_based
|   |   `-- density_estimation.py         # Clotho's reference-based adequacy score model (based on GMM)
|   |-- logprobs.py                       # Token-log-probability–based baseline metrics implementation
|   |-- sa.py                             # MDSA, base GMM metrics implementation
|   `-- semantic_entropy.py               # Semantic entropy baseline metric implementation
|-- models
|   |-- gemma.py                          # Interface wrapper for Gemma model for hidden state extraction
|   |-- llama.py                          # Interface wrapper for LLaMA model for hidden state extraction
|   `-- mistral.py                        # Interface wrapper for Mistral model for hidden state extraction
`-- preprocessing
    `-- feature_reduction.py              # PCA dimensionality reduction for hidden state feature vectors
```

## Instructions for Running Replication Experiments

### 1. Extracting hidden states from LLMs

To extract hidden states from LLMs, run the following command from the root of the repository:

```bash
$ cd data_extraction
$ HF_TOKEN="your_huggingface_token"
$ python 1-1_extract_hidden_states.py -t <task_name> -d <dataset_name> -p messages_template --model <model_name>
```

where `<task_name>` and `<dataset_name>` are defined in `clotho/exp_config.py` (e.g., `--task spell_check --dataset misspell_injected_wordnet`), and `<model_name>` is one of the supported models (`llama`, `gemma`, `mistral`).

You'll need to provide your own Hugging Face token as an environment variable. Additionally, you must first accept the terms of service for each model before accessing them via Hugging Face: [Llama](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Gemma](https://huggingface.co/google/gemma-2-9b-it), [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

### 2. Simulating Iterations of Clotho's Adequacy Modelling
To compute Clotho's pre-generation adequacy metrics using the extracted hidden states, run the following command from the root of the repository:

```bash
$ cd experiments
$ python run_clotho_iterations.py --target_task <task_name> --target_llm <model_name> --refset_extension_methods <reference_sampling_method> --seeds <seed1> <seed2> ...
```

where `<reference_sampling_method>` is one of `['random', 'diversity_euclidean', 'uncertainty', 'balanced']`.
