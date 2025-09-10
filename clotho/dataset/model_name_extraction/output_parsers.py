import json
import re
import ast

def parse_result_messages_template(result):
    if 'Extracted names:' not in result:
        return None, 'format_error'
    
    result = result.split('Extracted names:')[-1].strip()
    
    pattern = r'\[.*?\]'
    match = re.search(pattern, result)
    if match:
        parsed_result = match.group(0)
        try:
            parsed_result = ast.literal_eval(parsed_result)
        except Exception as e:
            return None, 'format_error'
        
        if not isinstance(parsed_result, list):
            return None, 'format_error'
        
        if len(parsed_result) == 0:
            return None, 'empty_list'
        
        # Check if all elements are strings
        for i, item in enumerate(parsed_result):
            if not isinstance(item, str):
                try:
                    parsed_result[i] = str(item)
                except Exception as e:
                    return None, 'non_string_item'
                
        return parsed_result, None

    else:
        return None, 'format_error'
    

template2parser = {
    'messages_template': parse_result_messages_template,
    'messages_template_refined': parse_result_messages_template,
}

def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    name_list, error = template2parser[template_name](actual_answer)
    expected_answer = expected_answer[0]
    
    if error:
        return False, error
    
    if len(expected_answer) == 1 and len(expected_answer[0]) == 0: 
        if name_list[0] == 'N/A':
            if return_parsed_answer:
                return True, name_list
            else:
                return True, None
        else:
            return False, 'no_names_expected'
    
    contains_all_required_names = True
    for name_group in expected_answer:
        if not any(name in name_list for name in name_group):
            contains_all_required_names = False
            break
            
    flattened_expected_names = [name for sublist in expected_answer for name in sublist]
    not_contains_unexpected_names = True
    for extracted_name in name_list:
        if extracted_name not in flattened_expected_names:
            not_contains_unexpected_names = False
            break
            
    if contains_all_required_names and not_contains_unexpected_names:
        if return_parsed_answer:
            return True, name_list
        else:
            return True, None
        
    semantic_errors = []
    if not contains_all_required_names:
        semantic_errors.append('missing_name')
    if not not_contains_unexpected_names:
        semantic_errors.append('unexpected_name')
    
    return False, ','.join(semantic_errors)
