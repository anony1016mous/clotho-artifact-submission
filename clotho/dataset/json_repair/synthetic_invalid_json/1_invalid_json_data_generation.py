# Fix types are borrowed from https://github.com/josdejong/jsonrepair

from openai import OpenAI
from dotenv import load_dotenv

import os
import re
import json

from tqdm import tqdm

load_dotenv()

client = OpenAI()

dataset_version = 2 # version 1 for `invalid_json_dataset_2166.jsonl`, version 2 for `invalid_json_dataset_4397.jsonl`


def generate_invalid_json_data(possible_fix, topic, N=10):
    user_message = f"""You are tasked to generate diverse test cases for a JSON repairing program. Generate {N} realistic and invalid JSON data with respect to a specific fix that needs to be applied. The data content should be related to the following topic: {topic}.

Possible Fix:
{possible_fix}

First, generate a valid and parsable JSON data containing diverse real-world content. Then, inject a specific formatting error to make it invalid JSON data but can be repaired by the given fix. The error should not alter the original content of the JSON data in a way that it is not fully recoverable by the given fix.

Ensure that after applying the given fix to the generated invalid JSON data, the result matches the original valid JSON data. For example, if the fix is 'Substitute Python None with JSON null' then any None values in the invalid JSON must correspond to null values in the original valid JSON. In other words, the invalid JSON data should be able to be repaired to the original valid JSON data without additional context or information.

Your answer should be in the following format (enclosing each JSON data in tags)
# Test Case 1
## Valid JSON Data:
<valid_json_data>
...valid JSON data here...
</valid_json_data>

## Invalid JSON Data with Injected Error:
<invalid_json_data>
...invalid JSON data here...
</invalid_json_data>

# Test Case 2
..."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful invalid JSON data generator."}
                    ]
                },
                {"role": "user",
                "content": [
                {"type": "text", "text": user_message},
                ],
            }],
            max_tokens=16384,
            temperature=0.8,
        )
    
    except Exception as e:
        print(f"Error: {e}")
        return None

    response_text = response.choices[0].message.content

    return response_text


def parse_test_cases(text):
    """
    주어진 text에서 (valid_json_data, invalid_json_data) string tuple의 리스트를 추출한다.
    """

    # Test Case 블록을 개별적으로 찾기
    test_case_pattern = re.compile(r'# Test Case \d+(.*?)(?=# Test Case \d+|$)', re.DOTALL)
    test_cases = test_case_pattern.findall(text)

    result = []
    for case in test_cases:
        # valid_json_data 추출
        valid_match = re.search(r'<valid_json_data>(.*?)</valid_json_data>', case, re.DOTALL)
        invalid_match = re.search(r'<invalid_json_data>(.*?)</invalid_json_data>', case, re.DOTALL)

        valid_json = valid_match.group(1).strip().strip("```json").strip("```") if valid_match else ''
        invalid_json = invalid_match.group(1).strip().strip("```json").strip("```") if invalid_match else ''
        
        if valid_json == '' or invalid_json == '':
            continue

        result.append((valid_json, invalid_json))

    return result

if __name__ == "__main__":
    N = 10
        
    os.makedirs(f"invalid_json_dataset", exist_ok=True)
    
    possible_fixes = [
        "Add missing quotes around keys",
        "Add missing escape characters",
        "Add missing commas",
        "Add missing closing brackets",
        "Repair truncated JSON",
        "Replace single quotes with double quotes",
        "Replace special quote characters like “...” with regular double quotes",
        "Replace special white space characters with regular spaces",
        "Replace Python constants None, True, and False with null, true, and false",
        "Strip trailing commas",
        "Strip comments like /* ... */ and // ...",
        "Strip fenced code blocks like ```json and ```",
        "Strip ellipsis in arrays and objects like [1, 2, 3, ...]",
        "Strip JSONP notation like callback({ ... })",
        "Strip escape characters from an escaped string like {\\\"stringified\\\": \\\"content\\\"}",
        "Strip MongoDB data types like NumberLong(2) and ISODate(\"2012-12-19T06:01:17.171Z\")",
        "Concatenate strings like \"long text\" + \"more text on next line\"",
        "Turn newline delimited JSON into a valid JSON array, for example: { \"id\": 1, \"name\": \"John\" } { \"id\": 2, \"name\": \"Sarah\" }",
    ]
    
    synthesized_possible_fixes = [
        "Fix unquoted string values\nExample: { \"key\": value } → { \"key\": \"value\" }",
        "Fix numeric values with leading zeros\nExample: { \"key\": 00123 } → { \"key\": 123 }",
        "Fix misplaced colons or equal signs\nExample: { key: value } → { \"key\": \"value\" }\nExample: { \"key\" = \"value\" } → { \"key\": \"value\" }",
        "Convert JavaScript object literals to valid JSON\nExample: { key: \"value\", method() {} } → { \"key\": \"value\" }",
        "Replace JavaScript undefined with null\nExample: { \"key\": undefined } → { \"key\": null }",
        "Replace NaN and Infinity with null\nExample: { \"key\": NaN } → { \"key\": null }",
        "Fix incorrect nesting (object inside array missing a bracket, etc.)",
        "Infer and add missing root brackets\nExample: ' \"key\": \"value\" ' → { \"key\": \"value\" }",
        "Handle concatenated JSON objects into a valid array\nExample: {...}{...} → [ {...}, {...} ]"
    ]

    topics_v1 = [
        "User profile information",
        "E-commerce order history",
        "Movie and TV show reviews",
        "Social media posts",
        "IoT sensor data",
        "Weather information",
        "Financial transaction records",
        "Health and fitness data",
        "Transportation and GPS location data",
        "Library book lending records"
    ]
    
    topics_v2 = [
        "Online course enrollment records",
        "Bank loan application data",
        "Customer service chat logs",
        "Smart home device usage logs",
        "Job application and resume data",
        "Airline flight booking records",
        "Restaurant reservation details",
        "Online gaming activity history",
        "Product inventory management data",
        "Digital subscription records",
        "Academic performance reports",
        "Real estate property listings",
        "Event ticket sales data",
        "Bug tracking and issue reports",
        "Vehicle maintenance history",
        "Employee attendance records",
        "Online forum discussion threads",
        "Retail store foot traffic data",
        "Music streaming playlists",
        "Public transportation card usage history"
    ]
    
    topics = topics_v1 if dataset_version == 1 else topics_v2
    
    for i, possible_fix in enumerate(tqdm(possible_fixes + synthesized_possible_fixes)):
        print(f"Generating invalid JSON data for fix: {possible_fix}")
        
        test_cases_parsed = []
        for topic_index in tqdm(range(len(topics))):
            generated_data = generate_invalid_json_data(possible_fix, topics[topic_index], N=N)
            
            if not generated_data:
                print(f"Failed to generate data for fix: {possible_fix}")
                continue
            
            test_cases = parse_test_cases(generated_data)
            for repaired, invalid in test_cases:
                try:
                    repaired_parsed = json.loads(repaired)  # Check if valid JSON
                except json.JSONDecodeError:
                    print(f"Invalid repaired JSON: {repaired}")
                    continue
                
                invokes_error = False
                try:
                    json.loads(invalid)  # Check if invalid JSON
                except json.JSONDecodeError:
                    invokes_error = True
                    
                if not invokes_error:
                    print(f"Invalid JSON did not invoke error: {invalid}")
                    continue
                
                test_cases_parsed.append((repaired_parsed, invalid.strip()))

            with open(f"invalid_json_dataset_v{dataset_version}/test_cases_{i+1}.json", "w") as f:
                json.dump({
                    "possible_fix": possible_fix,
                    "test_cases": test_cases_parsed
                }, f, indent=2)
        
        print(f"Generated {len(test_cases_parsed)} test cases for fix {i}")
        
        