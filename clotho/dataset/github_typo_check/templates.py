def messages_template(text_w_typo):
    return [
        {"role": "system", "content": f"""You are a helpful assistant."""},
        {"role": "user", "content": f"""Identify a typo in the following text from a GitHub repository:
---
{text_w_typo}
---

You are tasked to find only one typo (i.e., in a single word) in the text. Pinpoint the most likely typo.

Your answer must follow **exactly** this format:
Reasoning: <reason for the found typo (brief as in one line)>
Typo word: <typo>
Corrected word: <corrected_word>

Do not include any other explanation or extra output."""}]

template_map = {
    "messages_template": messages_template,
}