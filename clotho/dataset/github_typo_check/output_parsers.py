import re

def parse_result_messages_template(result):
    typo, correct = None, None
    for l in result.split('\n'):
        l = l.strip()
        if 'Typo word:' in l:
            typo = l.split('Typo word:')[1].strip()
        elif 'Corrected word:' in l:
            correct = l.split('Corrected word:')[1].strip()

    if typo is None or correct is None:
        return None, 'format_error'

    return (typo, correct), None


template2parser = {
    'messages_template': parse_result_messages_template,
}

def normalize(word):
    # remain only alphanumerics
    return re.sub(r'[^a-zA-Z0-9]', '', word)

def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    # answer key order: original, original_word, typo_word
    actual_parsed, error = template2parser[template_name](actual_answer)
    
    if error:
        return False, error
    
    typo, correct = actual_parsed
    
    typo_wo_special_chars = normalize(typo)
    if len(typo_wo_special_chars) > 0:
        typo = typo_wo_special_chars
        correct = normalize(correct)

    if typo == expected_answer[2] and correct == expected_answer[1]:
        if return_parsed_answer:
            return True, typo
        else:
            return True, None
    elif typo in expected_answer[2]:
        remaining_part = expected_answer[2].replace(typo, '<placeholder>')
        if remaining_part.replace('<placeholder>', correct) == expected_answer[1]:
            # print(f'Warning: typo "{typo}" and corrected word "{correct}" are substrings of expected answer "{expected_answer[2]}" and "{expected_answer[1]}" respectively.')
            if return_parsed_answer:
                return True, typo
            else:
                return True, None
        else:
            if return_parsed_answer:
                return False, typo
            else:
                return False, 'mismatch'
            
    else:
        if return_parsed_answer:
            return False, typo
        else:
            return False, 'mismatch'