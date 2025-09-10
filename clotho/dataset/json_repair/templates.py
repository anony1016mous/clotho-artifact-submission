# json repair tool: https://github.com/josdejong/jsonrepair?tab=readme-ov-file
# Compare with this?


def messages_template(input_dict):
    return [
        {"role": "system", "content": f"""You are an helpful assistant."""},
        {"role": "user", "content": f"""Your task is to repair the given invalid JSON data to be successfully parsed by a JSON parser.
Given JSON data:
{input_dict['invalid_json_str']}

Try to preserve the original content as much as possible while making it valid JSON. Find the minimal changes needed to make it valid JSON.

Your answer must follow **exactly** this format:
Why Invalid: <reason for the invalid JSON>
Suggested Repair: <suggested repair to the JSON data>
Repaired JSON: ```json
<repaired_json_data>
```"""}]


template_map = {
    "messages_template": messages_template,
}