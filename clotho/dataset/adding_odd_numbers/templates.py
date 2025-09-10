# source: https://www.promptingguide.ai/prompts/mathematics/odd-numbers

def messages_template(input_dict):
    return [
        {"role": "system", "content": f"""You are an helpful assistant."""},
        {"role": "user", "content": f"""Check whether the odd numbers in this group add up to an even number: {input_dict['sequence']}.
First, identify the odd numbers, add them, and indicate whether the result is odd or even.
Therefore, your answer must follow **exactly** this format:
Identified odd numbers: <list of identified odd numbers>
Sum of odd numbers: <sum of identified odd numbers (only the number)>
Result: <odd/even>"""}]

template_map = {
    "messages_template": messages_template,
}
