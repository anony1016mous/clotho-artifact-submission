import json

def parse_result_messages_template(result):
    if 'Repaired JSON:' not in result:
        return None, 'format_error'
    
    parsed_result = result.split('Repaired JSON:')[-1].strip()
    parsed_result = parsed_result.strip('```json'). strip('```')
    
    return parsed_result, None


template2parser = {
    'messages_template': parse_result_messages_template,
}

def normalize(word):
    # remain only alphanumerics
    return re.sub(r'[^a-zA-Z0-9]', '', word)

def check_json_equality(json1, json2):
    """
    Check if two JSON objects are equal, ignoring order of keys.
    """
    # FIXME: verify whether it also sorts nested keys
    return json.dumps(json1, sort_keys=True) == json.dumps(json2, sort_keys=True)

def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    repaired_json, error = template2parser[template_name](actual_answer)
    
    if error:
        return False, error
    
    try:
        repaired_json_parsed = json.loads(repaired_json)
    except json.JSONDecodeError:
        if return_parsed_answer:
            return False, repaired_json
        else:
            return False, 'json_decode_error'    
    
    if check_json_equality(repaired_json_parsed, expected_answer[0]):
        if return_parsed_answer:
            return True, repaired_json_parsed
        else:
            return True, None
        
    else:
        if return_parsed_answer:
            return False, repaired_json_parsed
        else:
            return False, 'content_mismatch'

if __name__ == "__main__":
    output_path = 'results/json_repair/inference_results/messages_template/invalid_json_dataset_2166_R10_T0.8.json'
    with open(output_path, 'r') as f:
        results = json.load(f)
        
    sample_data = results[0]
    
    is_correct, parsed_result = label_output(sample_data['inferences'][0], sample_data['GT'], 'messages_template', return_parsed_answer=True)
    print(f"Is the output correct? {is_correct}")
    print(f"Parsed result: {json.dumps(parsed_result, sort_keys=True)}")
    print(f"GT: {json.dumps(sample_data['GT'], sort_keys=True)}")
    
        
    