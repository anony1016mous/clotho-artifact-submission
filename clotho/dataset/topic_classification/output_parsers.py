import json

def parse_result_messages_template(result):
    if 'Category:' not in result:
        return None, 'format_error'
    
    parsed_result = result.split('Category:')[-1].strip()
    
    return parsed_result, None


template2parser = {
    'messages_template': parse_result_messages_template,
}

def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    category, error = template2parser[template_name](actual_answer)
    
    if error:
        return False, error
    
    if category == expected_answer[0]:
        if return_parsed_answer:
            return True, category
        else:
            return True, None
    else:
        if return_parsed_answer:
            return False, category
        else:
            return False, 'mismatch'
