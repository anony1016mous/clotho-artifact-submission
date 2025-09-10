# Fix types are borrowed from https://github.com/josdejong/jsonrepair

from openai import OpenAI
from dotenv import load_dotenv

import os
import re
import json

from tqdm import tqdm

load_dotenv()

client = OpenAI()


def generate_ML_paper_abstract(topic, complexity, N=10):
    user_message = f"""You are tasked to generate artificial ML paper abstracts that contains one or more specific ML model names to test a model name extraction program from a research paper abstract. Generate {N} realistic and diverse ML paper abstracts related to the following topic: {topic}. First, plan which specific ML model names to include in the abstract, then generate the abstract that contains those model names. The generated abstract should contain all of the planned model names, and should not contain any other model names that are not planned.
    
The complexity level of the abstract should be: {complexity}.
- simple: The abstract should be straightforward, with clear and concise language, suitable for a general audience.
- medium: The abstract should be moderately complex, using technical terms and concepts that are familiar to those with some background in machine learning.
- complex: The abstract should be highly technical, using advanced terminology and concepts that are typically understood by experts in the field of machine learning.

The names of ML models should indicate a unique and specific model (rather than a general class of models or architectures e.g., DNNs), such as "GPT-3", "Llama". Do not confuse with the following:
- Names of ML algorithms/methods rather than model names
- Names of ML tools or frameworks (e.g., "TensorFlow", "PyTorch", "Scikit-learn")
- Names of algorithmic frameworks for ML learning, training, or optimization (e.g., Expert Iteration (ExIt), AlphaZero, DAgger)
- Dataset names (e.g., "ImageNet", "CIFAR-10")
- Metrics, performance measures, or general statistical terms

Your answer should be in a valid JSON format as follows:
```json
{{
    "test_0": {{
        "model_names": ["<model_name_1>", ...],
        "abstract": "<abstract_text_here>"
    }},
    "test_1": ...
}}
```
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful research paper abstract generator as test data."}
                    ]
                },
                {"role": "user",
                "content": [
                {"type": "text", "text": user_message},
                ],
            }],
            temperature=0.8,
        )
    
    except Exception as e:
        print(f"Error: {e}")
        return None

    response_text = response.choices[0].message.content
    
    try:
        parsed_result = json.loads(response_text.strip('```json').strip('```'))
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Response text: {response_text}")
        return None
    
    for test_case_index, test_data in parsed_result.items():
        if not isinstance(test_data, dict) or 'model_names' not in test_data or 'abstract' not in test_data:
            print(f"Invalid test case format for {test_case_index}: {test_data}")
            return None
        
        model_names = test_data['model_names']
        abstract = test_data['abstract']
        
        if not isinstance(model_names, list) or not all(isinstance(name, str) for name in model_names):
            print(f"Invalid model names format for {test_case_index}: {model_names}")
            return None
        
        if not isinstance(abstract, str):
            print(f"Invalid abstract format for {test_case_index}: {abstract}")
            return None
        
        for model in model_names:
            if model not in abstract:
                print(f"Model name '{model}' not found in abstract for {test_case_index}. Abstract: {abstract}")
                return None

    return parsed_result


if __name__ == "__main__":
    N = 30

    os.makedirs(f"generations/ML_model_abstract", exist_ok=True)

    ml_categories = [
        "Model Architecture and Design",
        "Training Techniques and Optimization",
        "Explainability and Interpretability",
        "Fairness, Ethics, and Bias Mitigation",
        "Robustness and Adversarial Learning",
        "Transfer Learning and Domain Adaptation",
        "Resource-Efficient ML",
        "ML for Scientific Discovery",
        "Evaluation Metrics and Benchmarking",
        "Human-in-the-Loop and Interactive ML",
        "Self-Supervised and Unsupervised Learning",
        "Federated Learning and Privacy-Preserving ML",
        "Continual and Lifelong Learning",
        "Few-Shot and Zero-Shot Learning",
        "Multi-Modal Learning",
        "Reinforcement Learning and Policy Optimization",
        "Causal Inference in ML",
        "Graph Neural Networks and Relational Learning",
        "Time Series Forecasting and Sequential Models",
        "Natural Language Processing Models",
        "Computer Vision Models",
        "Audio and Speech Processing",
        "Generative Models (GANs, VAEs, Diffusion)",
        "Neuro-Symbolic and Hybrid AI",
        "AutoML and Neural Architecture Search",
        "Contrastive and Metric Learning",
        "Bayesian Methods and Uncertainty Quantification",
        "Out-of-Distribution Detection",
        "Synthetic Data Generation and Data Augmentation",
        "Scalable and Distributed Training",
        "ML for Healthcare and Biomedical Applications",
        "ML in Finance and Economics",
        "ML for Climate Science and Sustainability",
        "Anomaly Detection and Rare Event Modeling",
        "Model Calibration and Confidence Estimation",
        "Knowledge Distillation and Model Compression",
        "ML for Recommendation Systems",
        "ML for Robotics and Control",
        "Foundation Models and Large-Scale Pretraining",
        "Alignment and Safety of ML Systems"
    ]
        
    for i, topic in enumerate(tqdm(ml_categories)):
        print(f"Generating artificial abstracts w/ the topic of: {topic}")
        
        for j, complexity in enumerate(tqdm(['simple', 'medium', 'complex'])):
            print(f"Complexity level: {complexity}")
            test_data = None
            success = False
            for k in range(3):
                test_data = generate_ML_paper_abstract(topic, complexity, N=N)
            
                if test_data:
                    success = True
                    break

                else:
                    print(f"Failed to generate data for the topic: {topic}, Retrying... ({k+1}/3)")

            if not success:
                print(f"Failed to parse the generated data for topic: {topic} after multiple attempts.")
                continue
            
            with open(f"generations/ML_model_abstract/test_cases_{i+1}_{complexity}.json", "w") as f:
                json.dump(test_data, f, indent=2)
    
        print(f"Generated {len(test_data)} test cases for topic {i+1}/{len(ml_categories)}: {topic}")
        
        