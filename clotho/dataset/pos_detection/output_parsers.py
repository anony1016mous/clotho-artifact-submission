import json

pos_tags = [
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
    "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
    "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
    "VBZ", "WDT", "WP", "WP$", "WRB"
]

def parse_result_messages_template(result):
    if 'Part of Speech:' not in result:
        return None, 'format_error'
    
    parsed_result = result.split('Part of Speech:')[-1].strip()
    
    if "cantanswer" in parsed_result.lower():
        return 'CantAnswer', None
    
    if parsed_result not in pos_tags:
        return None, 'unknown_pos_tag'
    
    return parsed_result, None


template2parser = {
    'messages_template': parse_result_messages_template,
}

def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    predicted_pos_tag, error = template2parser[template_name](actual_answer)
    
    if error:
        return False, error
    
    if predicted_pos_tag == 'CantAnswer':
        if expected_answer[0] not in pos_tags:
            if return_parsed_answer:
                return True, predicted_pos_tag
            else:
                return True, None
        else:
            if return_parsed_answer:
                return False, predicted_pos_tag
            else:
                return False, 'cant_answer_but_valid_GT_present'
    
    if predicted_pos_tag == expected_answer[0]:
        if return_parsed_answer:
            return True, predicted_pos_tag
        else:
            return True, None
    else:
        if return_parsed_answer:
            return False, predicted_pos_tag
        else:
            return False, 'mismatch'
