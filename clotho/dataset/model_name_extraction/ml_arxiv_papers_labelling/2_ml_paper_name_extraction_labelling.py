# Fix types are borrowed from https://github.com/josdejong/jsonrepair

from openai import OpenAI
from dotenv import load_dotenv

import os
import re
import json

from tqdm import tqdm

load_dotenv()

client = OpenAI()


def identify_explicit_names(abstract):
    user_message = f"""Given the abstract of an ML (Machine Learning) research paper, extract all names of ML models explicitly mentioned in the abstract. An ML model is an object (stored locally in a file) that has been trained to recognize certain types of patterns.

### Include the following (pick):
- Names of ML models (e.g., "GPT-3", "Llama")

### Do not confuse with the following (do NOT pick):
- Names of ML algorithms/methods rather than model names
- Names of ML tools or frameworks (e.g., "TensorFlow", "PyTorch", "Scikit-learn")
- Names of algorithmic frameworks for ML learning, training, or optimization (e.g., Expert Iteration (ExIt), AlphaZero, DAgger)
- Dataset names (e.g., "ImageNet", "CIFAR-10")
- Metrics, performance measures, or general statistical terms

There may be multiple names with abbreviations, acronyms, or full names for the same concept. Group them together and return the list of the grouped names.

Abstract:
{abstract} 

Your response MUST be in the following format:
### Grouping of extracted names and justification (one name group per line)
```jsonl
["<name1>"]
["<name2-1>", "<name2-2>", ...]
...```

If you find no names, return an empty list like the following:
```jsonl
[]
```"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful model name extractor in a research paper abstract."}
                    ]
                },
                {"role": "user",
                "content": [
                {"type": "text", "text": user_message},
                ],
            }],
            max_tokens=4096,
            temperature=0.8,
        )
    
    except Exception as e:
        print(f"Error: {e}")
        return None

    response_text = response.choices[0].message.content
    
    pattern = r'```jsonl\s*(.*?)\s*```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    if len(matches) == 0:
        print(f"No JSONL block found in the response:\n{response_text}")
        return None
    
    final_verified_names = []
    for l in matches[0].splitlines():
        l = l.strip()
        if not l:
            continue
        try:
            final_verified_names.append(json.loads(l))
        except json.JSONDecodeError:
            print(f"Invalid JSON in response: {l}")
            return None

    return final_verified_names


if __name__ == "__main__":
    N = 5
    
    with open("targets/ml_arxiv_papers_raw_no_labels.jsonl", "r") as f:
        papers = [json.loads(line) for line in f.readlines()]

    dataset_with_labels = []
    conflicting_model_names_count = 0
    for i, paper_info in enumerate(tqdm(papers)):
        abstract = paper_info["abstract"]
        
        responses = []
        for j in range(N):
            names = identify_explicit_names(abstract)
            if not names:
                print(f"Failed to generate answer for the paper {j}")
                continue
            
            responses.append(names)
            
        if len(responses) == 0:
            print(f"Failed to generate any names for the paper {i}")
            continue
            
        names_reps = []
        for response in responses:
            names_sorted = []
            for name_group in response:
                name_group_str = ",".join(sorted(name_group))
                names_sorted.append(name_group_str)
                
            names_reps.append(",".join(sorted(names_sorted)))
            
        if len(set(names_reps)) != 1:   # has conflicting names
            dataset_with_labels.append({
                "title": paper_info["title"],
                "abstract": abstract,
                "model_names": responses,
                "has_conflicting_names": True
            })
            conflicting_model_names_count += 1
            
        else:
            dataset_with_labels.append({
                "title": paper_info["title"],
                "abstract": abstract,
                "model_names": responses[0],
            })
            
        with open(f"generations/ml_arxiv_papers_gpt4_labelled_N5.json", "w") as f:
            json.dump(dataset_with_labels, f, indent=4)
    
    print("Total papers with conflicting model names:", conflicting_model_names_count)
    print("Total papers processed:", len(dataset_with_labels))
