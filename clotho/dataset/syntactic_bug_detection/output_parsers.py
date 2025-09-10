bug_types = [
    'missing_colon',
    'missing_parenthesis',
    'missing_quotation',
    'missing_comma',
    'mismatched_quotation',
    'mismatched_bracket',
    'keywords_as_identifier'
]

def parse_result_messages_template_refined(result):
    line_number, bug_type = None, None
    answer_line = None
    for l in result.split('\n'):
        l = l.strip()
        if 'Answer:' in l:
            answer_line = l.split('Answer:')[1].strip()
            break

    if answer_line is None:
        return None, 'format_error'
    
    splitted = answer_line.split(',')
    if len(splitted) != 2:
        return None, 'format_error'
    
    try:
        line_number = int(splitted[0].strip())
        bug_type = splitted[1].strip()
    except:
        return None, 'format_error'

    if bug_type not in bug_types:
        return None, 'bug_type_error'

    return (line_number, bug_type), None


def parse_result_messages_template(result):
    line_number, bug_type = None, None
    answer_line = None
    for l in result.split('\n'):
        l = l.strip()
        if 'Answer:' in l:
            answer_line = l.split('Answer:')[1].strip()
            break

    if answer_line is None:
        return None, 'format_error'
    
    splitted = answer_line.split(',')
    if len(splitted) != 2:
        return None, 'format_error'
    
    try:
        line_number = int(splitted[0].strip())
        bug_type = splitted[1].strip()
    except:
        return None, 'format_error'

    if bug_type not in bug_types:
        return None, 'bug_type_error'

    return (line_number, bug_type), None



def parse_result_messages_template_handling_exceptional_cases(result):
    bugs = []
    has_format_error = False
    
    for l in result.split('\n'):
        l = l.strip()
        
        bug_result = l.split(',')
        if len(bug_result) != 3:
            has_format_error = True
            continue
        
        try:
            line_number = int(bug_result[0].strip())
        except:
            has_format_error = True
            continue
        bug_type = bug_result[1].strip()
        
        bugs.append((line_number, bug_type))
        
    return bugs, 'format_error' if has_format_error else None
        
        
template2parser = {
    'messages_template': parse_result_messages_template,
}


def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    actual_parsed, error = template2parser[template_name](actual_answer)
    
    if error:
        return False, error
    
    line_number, bug = actual_parsed

    if line_number == int(expected_answer[0]) and bug == expected_answer[1]:
        line_number, bug = actual_parsed
        if return_parsed_answer:
            return True, (line_number, bug)
        else:
            return True, None
    else:
        if return_parsed_answer:
            return False, (line_number, bug)
        else:
            return False, 'mismatch'