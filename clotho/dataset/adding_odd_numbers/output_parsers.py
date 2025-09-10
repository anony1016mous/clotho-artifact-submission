import json
import re

def parse_result_messages_template(result):
    result = result.lower()
    if 'sum of odd numbers:' not in result and 'sum of identified odd numbers:' not in result:
        return (None, None), 'format_error_no_sum'
    
    if 'result:' not in result:
        return (None, None), 'format_error_no_results'
    
    pattern = r'sum of (identified )?odd numbers:\s*(\d+)'
    match = re.search(pattern, result)
    if not match:
        return (None, None), 'format_error'

    parsed_result = match.group(2)
    
    if not parsed_result.isdigit():
        return (None, None), 'format_error'
    
    # check final result (odd/even)
    choice = result.split('result:')[-1].strip()
    
    if 'odd' in choice and 'even' in choice:
        return (None, None), 'format_error'
    
    if 'odd' in choice:
        choice = 'odd'
    if 'even' in choice:
        choice = 'even'

    return (int(parsed_result), choice), None


template2parser = {
    'messages_template': parse_result_messages_template,
}

def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    (sum_of_odds, choice), error = template2parser[template_name](actual_answer)
    
    if error:
        return False, error
    
    GT_choice = 'even' if expected_answer[0] % 2 == 0 else 'odd'

    if sum_of_odds == expected_answer[0]:
        if choice == GT_choice:
            if return_parsed_answer:
                return True, sum_of_odds
            else:
                return True, None
        else:
            return False, 'odd_even_confusion'
    else:
        if return_parsed_answer:
            return False, sum_of_odds
        else:
            return False, 'mismatch'
