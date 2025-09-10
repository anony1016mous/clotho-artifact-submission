import re

def parse_result_messages_template(result):
    misspelled, correct = None, None
    for l in result.split('\n'):
        l = l.strip()
        if 'Misspelled word:' in l:
            misspelled = l.split('Misspelled word:')[1].strip()
        elif 'Corrected word:' in l:
            correct = l.split('Corrected word:')[1].strip()

    if misspelled is None or correct is None:
        return None, 'format_error'

    return (misspelled, correct), None


template2parser = {
    'messages_template': parse_result_messages_template,
}

def normalize(word):
    # remain only alphabets 
    return re.sub(r'[^a-zA-Z]', '', word)
    

def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    # answer key order: original, original_word, misspelled_word
    actual_parsed, error = template2parser[template_name](actual_answer)
    
    if error:
        return False, error
    
    misspelled, correct = actual_parsed

    # FIXME: does not check the validity of the corrected word (there can be multiple valid corrections)
    if normalize(misspelled) == normalize(expected_answer[2]):
        if return_parsed_answer:
            return True, misspelled
        else:
            return True, None
    else:
        if return_parsed_answer:
            return False, misspelled
        else:
            return False, 'mismatch'