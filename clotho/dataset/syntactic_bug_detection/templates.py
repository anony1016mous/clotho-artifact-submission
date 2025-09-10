
def messages_template(input_text):
    return [
        {"role": "system", "content": f"""You are a skillful syntactic bug detector."""},
        {"role": "user", "content": f"""You will be given a Python code snippet that contains a **syntactic bug**. Your task is to identify the **line number** where the bug occurs and the **type of syntactic bug**. Choose the bug type from the list below:

**Bug types and their meanings:**
- missing_colon: A required colon is missing
- missing_parenthesis: A parenthesis (or bracket or brace) is opened but not properly closed
- mismatched_bracket: brackets, parenthesis, or braces around a string do not match
- missing_quotation: One of the quotes around a string is missing
- mismatched_quotation: Quotes around a string do not match
- missing_comma: A required comma is missing
- keywords_as_identifier: A reserved Python keyword is used as a variable or function name

**Code with a syntactic bug (only one line contains a bug):**
{input_text}

Your answer must follow **exactly** this format:  
Reasoning: <reason for the bug>
Answer: <line_number>, <bug_type>
Do not include any explanation or extra output."""}]

# ------------------------------------------------------------------- prompt variations for exploratory purposes

def messages_template_handling_exceptional_cases(input_text):
    # Consider multiple syntactic bugs (but it seems that the performance is significantly degraded by using this template)
    return [
        {"role": "system", "content": f"""You are a skillful syntactic bug detector."""},
        {"role": "user", "content": f"""You will be given a code snippet that may contain **syntactic bugs**. Your task is to identify **all lines** where bugs occur and their **bug types**. The code might have a single bug, multiple bugs or no bugs at all.


**Known bug types**:
missing_colon, missing_parenthesis, missing_quotation, missing_comma, mismatched_quotation, mismatched_bracket, keywords_as_identifier

You may also identify other types of syntactic bugs beyond this list if you encounter them. In that case, name the bug type appropriately in snake_case and explain it in the reasoning part.

**Code to analyze:**
{input_text}

Your answer must follow these formats depending on the number of bugs found (do not include any extra text rather than the formatted answer):
A. If a single bug is found, your answer should be in the following format:
<line_number>, <bug_type>, <reasoning>

B. If multiple bugs are found, your answer should be in the following format:
<line_number_1>, <bug_type_1>, <reasoning_1>
<line_number_2>, <bug_type_2>, <reasoning_2>
...
  
C. If no bugs are found, your answer should be:
N/A"""}]


def messages_template_minimal(input_text):
    return [
        {"role": "system", "content": f"""You are a helpful assistant."""},
        {"role": "user", "content": f"""Find a syntactic bug in the following code:
{input_text}"""}]


def messages_template_formatdiff_1(input_text):
    return [
        {"role": "system", "content": f"""You are a skillful syntactic bug detector."""},
        {"role": "user", "content": f"""**Code with a syntactic bug:**
{input_text}

**Potential bug types:**
missing_colon
missing_parenthesis
missing_quotation
missing_comma
mismatched_quotation
mismatched_bracket
keywords_as_identifier

Your answer must be in the following format: <line_number>, <bug_type>
Do not generate anything else."""}]


def messages_template_formatdiff_2(input_text):
    return [
        {"role": "system", "content": "You are a skillful syntactic bug detector."},
        {"role": "user", "content": f"""Analyze the following code:
{input_text}

Identify any syntactic bug and classify it as one of:
- missing_colon
- missing_parenthesis
- missing_quotation
- missing_comma
- mismatched_quotation
- mismatched_bracket
- keywords_as_identifier

Provide the result in this format:
<line_number>, <bug_type>

Strict Rule: Only return the answer in the exact format specified. Do not include explanations."""}
]

def messages_template_formatdiff_3(input_text):
    return [
        {"role": "system", "content": "You analyze code and identify syntactic errors."},
        {"role": "user", "content": f"""Inspect the following code for syntax issues:
{input_text}

Identify any syntactic bug and classify it as one of:
- missing_colon
- missing_parenthesis
- missing_quotation
- missing_comma
- mismatched_quotation
- mismatched_bracket
- keywords_as_identifier

Provide the result in this format:
<line_number>, <bug_type>

Strict Rule: Only return the answer in the exact format specified. Do not include explanations."""}
]


def messages_template_formatdiff_4(input_text):
    return [
        {"role": "system", "content": "You are a skillful syntactic bug detector."},
        {"role": "user", "content": f"""### Code to Analyze:

{input_text}

| Potential Bug Type |
|--------------------|
| missing_colon      |
| missing_parenthesis|
| missing_quotation  |
| missing_comma      |
| mismatched_quotation |
| mismatched_bracket |
| keywords_as_identifier |

**Expected Response Format:** `<line_number>, <bug_type>`

**Note:** Return only the formatted answer. Do not include extra text."""}]

def messages_template_formatdiff_5(input_text):
    return [
        {"role": "system", "content": "You are responsible for verifying syntax in Python code."},
        {"role": "user", "content": f"""### Syntax Bug Detection Test

#### Code Sample:
```python
{input_text}
```
Check for:

✅ missing_colon
✅ missing_parenthesis
✅ missing_quotation
✅ missing_comma
✅ mismatched_quotation
✅ mismatched_bracket
✅ keywords_as_identifier

Expected Output:

<line_number>, <bug_type>

❗Important: Do not include explanations, only output in the required format."""}]

def messages_template_taskdiff_summarize(input_text):
    return [
        {"role": "system", "content": f"""You are a code summarizer."""},
        {"role": "user", "content": f"""**Code to summarize:**
{input_text}

Your answer must include a brief summary of the code. Do not generate anything else."""}]


def messages_template_taskdiff_comment_gen(input_text):
    return [
        {"role": "system", "content": f"""You are a code comment generator."""},
        {"role": "user", "content": f"""**Function to add a comment:**
{input_text}

Generate a comment describing the function's behavior. Your answer must include only the comment. Do not generate anything else."""}]


def messages_template_taskdiff_test_gen(input_text):
    return [
        {"role": "system", "content": f"""You are a unit test generator."""},
        {"role": "user", "content": f"""**Code Under Test:**
{input_text}

Generate a unit test for the given code in Pytest format. Your answer must include a single test function. Do not generate anything else."""}]


def messages_template_simtask_syntactic_checker(input_text):
    return [
        {"role": "system", "content": f"""You are a Python interpreter."""},
        {"role": "user", "content": f"""**Code to check:**
{input_text}

Generate an error message as you are an interpreter. Give the error message that would be produced when the code is run, or, if the code is correct, answer with [No error]. Do not generate anything else."""}]
    
    
def messages_template_original(input_text):
    # Use the same prompt in the original paper: https://github.com/HammingHQ/bug-in-the-code-stack/blob/main/notebooks/bug_in_the_code_stack_experiment_litellm_meta_llama3.ipynb

    return [
        {"role": "system", "content": f"""You are a helpful assistant who can detect a syntactic bug in code."""},
        {"role": "user", "content": f"""I will give you a codebase with a syntactic bug hidden at some line. You need to answer the line at which the syntactic error occurs and the type of the syntactic error.

<example>
1 | def fahrenheit_to_celsius(fahrenheit):
2 |   return (fahrenheit - 32) * 5.0/9.0
3 |
4 | def is_prime(num:
5 |     if num <= 1:
6 |         return False
7 |     for i in range(2, int(num**0.5) + 1):
8 |         if num % i == 0:
9 |             return False
10|     return True
Answer: 4, missing_parenthesis
</example>

<example>
1| import random
2| import string
3|
4| def generate_password(length=8):
5|     characters = string.ascii_letters + string.digits
6|     password = '\".join(random.choice(characters) for i in range(length))
7|     return password
Answer: 6, mismatched_quotation
</example>

<context>
{input_text}
</context>

<bug_types>
missing_colon
missing_parenthesis
missing_quotation
missing_comma
mismatched_quotation
mismatched_bracket
keywords_as_identifier
</bug_types>

Always return your answer in the following format: <line_number>, <bug_type>
Do not write anything else after that."""}]
    
template_map = {
    "messages_template": messages_template,
}