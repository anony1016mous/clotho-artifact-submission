def messages_template(input_dict):
    return [
        {"role": "system", "content": f"""You are a helpful assistant that can determine the part of speech of a word in a sentence."""},
        {"role": "user", "content": f"""In this task, you will be presented with a sentence and a word contained in that sentence. You have to determine the part of speech for a given word and return just the tag for the word's part of speech. 

Return only the part of speech tag. If the word cannot be tagged with the listed tags, return Unknown. If you are unable to tag the word, return CantAnswer.

Here is the Alphabetical list of part-of-speech tags used in this task: CC: Coordinating conjunction, CD: Cardinal number, DT: Determiner, EX: Existential there, FW: Foreign word, IN: Preposition or subordinating conjunction, JJ: Adjective, JJR: Adjective, comparative, JJS: Adjective, superlative, LS: List item marker, MD: Modal, NN: Noun, singular or mass, NNS: Noun,
plural, NNP: Proper noun, singular, NNPS: Proper noun, plural, PDT: Predeterminer, POS: Possessive ending, PRP: Personal pronoun, PRP$: Possessive pronoun, RB: Adverb, RBR: Adverb, comparative, RBS: Adverb, superlative, RP: Particle, SYM: Symbol, TO: to, UH: Interjection, VB: Verb, base form, VBD: Verb, past tense, VBG: Verb, gerund or present participle, VBN: Verb,
past participle, VBP: Verb, non-3rd person singular present, VBZ: Verb, 3rd person singular present, WDT: Wh-determiner, WP: Wh-pronoun, WP$: Possessive wh-pronoun, WRB: Wh-adverb

Sentence: {input_dict['sentence']}
Word: {input_dict['token']}

Your answer must follow **exactly** this format:
Reasoning: <your reasoning for the part of speech tag>
Part of Speech: <the part of speech tag for the word>"""}]

# tool calling for llama3.1 model: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling

template_map = {
    "messages_template": messages_template,
}
