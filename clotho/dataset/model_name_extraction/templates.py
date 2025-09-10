def messages_template(input_dict):
    return [
        {"role": "system", "content": f"""You are a helpful assistant."""},
        {"role": "user", "content": f"""Given the abstract of a research paper, extract all names of ML models.
         
Do not include names of algorithms, tools, or datasets. If there are both full names and abbreviations, include one of them.

Abstract:
{input_dict['abstract']}

Your answer must follow **exactly** this format (do not include any other explanation or extra output):
Reasoning: <Your reasoning for the extraction>
Extracted names: ["<name_1>", "<name_2>", ...]

If you don't find any explicit names in the abstract, return ["N/A"] instead as the extracted names list."""}
    ]


def messages_template_refined(input_dict):
    return [
        {"role": "system", "content": f"""You are a helpful assistant."""},
        {"role": "user", "content": f"""Given the abstract of a research paper, extract all names of ML models explicitly mentioned in the text. (An ML model is an object (stored locally in a file) that has been trained to recognize certain types of patterns)

Do not include names of general model types or categories, or architectures.
Do not include names of algorithms, tools, or datasets. If there are both full names and abbreviations, include one of them.

Abstract:
{input_dict['abstract']}

Your answer must follow **exactly** this format (do not include any other explanation or extra output):
Reasoning: <Your reasoning for the extraction>
Extracted names: ["<name_1>", "<name_2>", ...]

If you don't find any explicit names in the abstract, return ["N/A"] instead as the extracted names list."""}
    ]


template_map = {
    "messages_template": messages_template,
    "messages_template_refined": messages_template_refined
}
