# https://pupuweb.com/how-use-llama-2-text-classification-tasks/

def messages_template(input_dict):
    return [
        {"role": "system", "content": f"""You are a helpful assistant that can classify news articles into categories."""},
        {"role": "user", "content": f"""In this task, you will be presented with a news article. A news article can be classified as one of the following categories: World, Sports, Business, Sci/Tech.

If the given article does not fit into any of these categories, you should classify it as "Other".

Examples:
- World: "UN chief urges action on climate change as report warns of 'catastrophe'"
- Sports: "Ronaldo scores twice in Manchester United return"
- Business: "Apple delays plan to scan iPhones for child abuse images"
- Sci/Tech: "SpaceX launches first all-civilian crew into orbit"

Based on these categories, classify the given news article.
Article: {input_dict['article']}

Your answer must follow **exactly** this format:
Reasoning: <your reasoning for the category>
Category: <category of the article (World, Sports, Business, Sci/Tech, Other)>"""}]


template_map = {
    "messages_template": messages_template,
}
