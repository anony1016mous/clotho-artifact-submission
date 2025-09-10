import clotho.dataset as clotho_dataset

supported_datasets = {
    'syntactic_bug_detection': ['syntactic_bug_injected'],
    'spell_check': ['misspell_injected_wordnet'],
    'github_typo_check': ['github_typo_corpus_cleaned'],
    'json_repair': ['invalid_json_dataset_2166', 'invalid_json_dataset_4397'],
    'pos_detection': ['cleaned_and_sampled_pos_tags', 'cleaned_and_sampled_pos_tags_trainset'],
    'adding_odd_numbers': ['integer_sequences_length_1_to_10_uniform'],
    'model_name_extraction': ['ml_arxiv_papers_no_conflicting_labels', 'synthetic_abstracts_gpt4o_3600'],
    'topic_classification': ['ag_news_test']
}

task2template = {
    'syntactic_bug_detection': clotho_dataset.syntactic_bug_detection.templates.template_map,
    'spell_check': clotho_dataset.spell_check.templates.template_map,
    'github_typo_check': clotho_dataset.github_typo_check.templates.template_map,
    'json_repair': clotho_dataset.json_repair.templates.template_map,
    'pos_detection': clotho_dataset.pos_detection.templates.template_map,
    'adding_odd_numbers': clotho_dataset.adding_odd_numbers.templates.template_map,
    'model_name_extraction': clotho_dataset.model_name_extraction.templates.template_map,
    'topic_classification': clotho_dataset.topic_classification.templates.template_map
}

task2labeler = {
    'syntactic_bug_detection': clotho_dataset.syntactic_bug_detection.output_parsers.label_output,
    'spell_check': clotho_dataset.spell_check.output_parsers.label_output,
    'github_typo_check': clotho_dataset.github_typo_check.output_parsers.label_output,
    'json_repair': clotho_dataset.json_repair.output_parsers.label_output,
    'pos_detection': clotho_dataset.pos_detection.output_parsers.label_output,
    'adding_odd_numbers': clotho_dataset.adding_odd_numbers.output_parsers.label_output,
    'model_name_extraction': clotho_dataset.model_name_extraction.output_parsers.label_output,
    'topic_classification': clotho_dataset.topic_classification.output_parsers.label_output
}

task2input_key = {
    'syntactic_bug_detection': 'code',
    'spell_check': 'misspelled',
    'github_typo_check': 'misspelled',
    'json_repair': None, # give the whole input dict (supporting multiple variables),
    'pos_detection': None,
    'adding_odd_numbers': None,
    'model_name_extraction': None,
    'topic_classification': None
}

task2template_keywords = {
    'syntactic_bug_detection': ['Answer:'],
    'spell_check': ['Misspelled word:', 'Corrected word:'],
    'github_typo_check': ['Reasoning:', 'Typo word:', 'Corrected word:'],
    'json_repair': ['Why Invalid:', 'Suggested Repair:', 'Repaired JSON:'],
    'pos_detection': ['Reasoning:', 'Part of Speech:'],
    'adding_odd_numbers': ['Identified odd numbers:', 'Sum of odd numbers:', 'Result:'],
    'model_name_extraction': ['Reasoning:', 'Extracted names:'],
    'topic_classification': ['Reasoning:', 'Category:']
}

task2answer_key_list = {
    'syntactic_bug_detection': ['line_number', 'bug_type'],
    'spell_check': ['original', 'original_word', 'misspelled_word'],
    'github_typo_check': ['original', 'original_word', 'misspelled_word'],
    'json_repair': ['repaired_json'],
    'pos_detection': ['upos'],
    'adding_odd_numbers': ['sum_of_odds'],
    'model_name_extraction': ['model_names'],
    'topic_classification': ['topic']
}
