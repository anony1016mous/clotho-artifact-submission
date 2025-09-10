from datasets import load_dataset
import random
import json
random.seed(42)

ds = load_dataset("CShorten/ML-ArXiv-Papers")
dataset = []

for item in ds["train"]:
    dataset.append({
        "title": item["title"],
        "abstract": item["abstract"].strip()
    })
    

dataset = random.sample(dataset, 50000)
with open("ml_arxiv_papers_raw_no_labels.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")
