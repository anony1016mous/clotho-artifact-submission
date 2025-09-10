import json
import glob

generated_data_path = 'generations/invalid_json_dataset_v2'

all_test_cases = []

def flatten(obj):
    results = []
    
    def _flatten(o):
        if isinstance(o, dict):
            for k, v in o.items():
                results.append(k)
                _flatten(v)
        elif isinstance(o, list):
            for item in o:
                _flatten(item)
        elif o is None or (o == True or o == False):
            pass
        else:
            results.append(str(o))
    
    _flatten(obj)
    return results

for test_case_file in glob.glob(f'{generated_data_path}/*.json'):
    with open(test_case_file, 'r') as f:
        test_case_data = json.load(f)
        
    for repaired_json, invalid_json_str in test_case_data["test_cases"]:
        # Check if all the keys and the values in the repaired JSON are present in the invalid JSON string
        leafs = flatten(repaired_json)
        
        recoverable = True
        missing_leaf = []
        for s in leafs:
            if s is None:
                continue
            if str(s) not in invalid_json_str:
                recoverable = False
                missing_leaf.append(s)
                break
            
        if not recoverable:
            print(f'Possibly not recoverable (missing: {missing_leaf}):\n{json.dumps(repaired_json, indent=2)}\n{invalid_json_str}\n---\n')
            continue
        
        all_test_cases.append({
            "repaired_json": repaired_json,
            "invalid_json_str": invalid_json_str,
            "fix_type": test_case_data["possible_fix"]
        })
        
with open('generations/invalid_json_dataset_4397.jsonl', 'w') as f:
    for test_case in all_test_cases:
        f.write(json.dumps(test_case) + '\n')