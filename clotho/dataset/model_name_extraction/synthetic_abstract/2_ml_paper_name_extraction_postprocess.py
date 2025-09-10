import json
import glob
import re

data_path = 'ML_model_abstract/'

dataset = []

def extract_full_model_name(abstract, abbreviation):
    pattern = r'([\w\s\-]+?)\s*\(\s*' + re.escape(abbreviation) + r'\s*\)'
    match = re.search(pattern, abstract)
    
    if match:
        possible_full_name = match.group(1).strip()
        words = possible_full_name.split()
        
        if len(words) >= len(abbreviation):
            candidate_words = words[-len(abbreviation):]
            initials = ''.join(word[0].upper() for word in candidate_words)
            
            if initials == abbreviation.upper():
                return ' '.join(candidate_words)
    
    return None

for file in glob.glob(data_path + '*.json'):
    with open(file, 'r') as f:
        data = json.load(f)
    
    for test_id, test_case in data.items():
        model_names = test_case['model_names']
        
        model_name_groups = []
        
        for model_name in model_names:
            name_group = []
            model_name = model_name.strip()
            
            abbreviation = None
            last_word = None
            full_model_name = None
            # Extract the possible abbreviation enclosed in parentheses
            match = re.search(r'\((.*?)\)', model_name)
            if match:
                abbreviation = match.group(1).strip()

            model_name_split = model_name.split()
            if len(model_name_split) > 1:
                if 'OpenAI' in model_name or 'Google' in model_name or 'DeepMind' in model_name:
                    last_word = model_name_split[-1].strip()
                    
            match = re.search(re.escape(model_name) + r'\s*\((.*?)\)', test_case['abstract'])
            if match:
                abbreviation = match.group(1).strip()
            
            if abbreviation:
                full_model_name = extract_full_model_name(test_case['abstract'], abbreviation)
            else:
                full_model_name = extract_full_model_name(test_case['abstract'], model_name)
            
            name_group.append(model_name)
            if abbreviation:
                name_group.append(abbreviation)
            if last_word:
                name_group.append(last_word)
            if full_model_name:
                name_group.append(full_model_name)
                
            for name in name_group:
                assert name in test_case['abstract'], f"Model name '{name}' not found in abstract for test case {test_id}"
                
            model_name_groups.append(name_group)
            
        dataset.append({
            'model_names': model_name_groups,
            'abstract': test_case['abstract'],
        })


with open('synthetic_abstracts_gpt4o_3600.jsonl', 'w') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + '\n')