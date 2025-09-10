import json

# Only select models that did not have any conflicting name labellings by GPT-4

dataset = []

with open('ml_arxiv_papers_gpt4_labelled_N5.json') as f:
    dataset = json.load(f)
    
targets = []
targets_with_names = []
targets_no_names = []

for item in dataset:
    if 'has_conflicting_names' not in item:
        if len(item['model_names']) == 0 or (len(item['model_names']) == 1 and len(item['model_names'][0]) == 0):
            targets_no_names.append(item)
        else:
            targets_with_names.append(item)
        targets.append(item)
        
print(f"Total items: {len(dataset)}")
print(f"Items with no conflicting names: {len(targets)}")
print(f"Items with names: {len(targets_with_names)}")
        
selected_targets = targets_with_names + targets_no_names[:3000]

with open('ml_arxiv_papers_no_conflicting_labels.jsonl', 'w', encoding='utf-8') as f:
    for item in selected_targets:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
print(f"Total selected items: {len(selected_targets)}")
