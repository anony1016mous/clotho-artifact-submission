import dotenv
dotenv.load_dotenv()
import os


import clotho.inference as inference
from collections import defaultdict
from tqdm import tqdm

import json
import torch
import numpy as np

# Start from an initial test cases (10 manually crafted "typical" test inputs for now)

initial_tests = {
    'syntactic_bug_detection': [
        {
            "code": "1| for i in range(1 len(data)): 2|     print(i)",  # 1
            "label": [1, "missing_comma"]
        },
        {
            "code": "1| def continue(self):\n2|     pass",  # 2
            "label": [1, "keywords_as_identifier"]
        },
        {
            "code": "1| raise Exception('Error']",  # 3
            "label": [1, "mismatched_bracket"],
        },
        {
            "code": "1| a = 1\n2| result = a\n3| return (result True)",  # 4
            "label": [3, "missing_comma"]
        },
        {
            "code": "1| def func(x)\n2|     x += 1\n3|     return x",  # 5
            "label": [1, "missing_colon"],
        },
        {
            "code": "1| return \"Hello World",  # 6
            "label": [1, "missing_quotation"]
        },
        {
            "code": "1| x = input(\"Enter a number: \')",   # 7
            "label": [1, "mismatched_quotation"]
        },
        {
            "code": "1| def = False\n2| print(None)",  # 8
            "label": [1, "keywords_as_identifier"]
        },
        {
            "code": "1| if x > 10:\n2|     print('x is greater than 10')\n3| else\n4|     print('x is less than or equal to 10')",  # 9
            "label": [3, "missing_colon"]
        },
        {
            "code": "1| for i in range(5:\n2|     print(i)",  # 10
            "label": [1, "missing_parenthesis"]
        }
    ],
    'topic_classification': [
        {
            "article": "Federal Reserve raises interest rates for the third time this year in an effort to curb inflation.",
            "label": ["Business"]
        },
        {
            "article": "The Coastal City Strikers secured a 2-1 victory in the national cup semifinal, advancing to their first championship match in over a decade.",
            "label": ["Sports"]
        },
        {
            "article": "NASA confirmed the successful deployment of the Lunar Surface Habitat prototype during its Artemis IV mission simulation. Engineers say the inflatable structure could support up to four astronauts for extended stays on the Moon.",
            "label": ["Sci/Tech"]
        },
        {
            "article": "The United Nations Security Council convened an emergency session after renewed clashes along the Armenia-Azerbaijan border left at least 14 dead. Diplomats from multiple nations called for an immediate ceasefire, warning that the violence risked destabilizing the broader Caucasus region.",
            "label": ["World"]
        },
        {
            "article": "Researchers at the University of Toronto unveiled a breakthrough AI model capable of accurately predicting protein folding for previously uncharacterized enzymes.",
            "label": ["Sci/Tech"]
        },
        {
            "article": "Global markets responded with cautious optimism after the U.S. administration announced a temporary halt to planned tariffs on Chinese imports. Analysts noted that while the pause could ease immediate supply chain pressures, uncertainty remains over whether this represents a long-term shift in trade strategy or merely a short-term political maneuver.",
            "label": ["Business"]
        },
        {
            "article": "Japan and South Korea announced a landmark defense pact in Seoul, pledging to share real-time intelligence on missile launches from North Korea.",
            "label": ["World"]
        },
        {
            "article": "The Metro City Mariners announced plans for a $250 million renovation of their home stadium, which will include expanded seating, upgraded training facilities, and a retractable roof to accommodate year-round events.",
            "label": ["Sports"]
        },
        {
            "article": "Private space firm AstraNova successfully tested a reusable rocket stage that can be turned around for launch within 24 hours. Engineers credit a new ceramic heat shield design for withstanding repeated reentries without significant degradation.",
            "label": ["Sci/Tech"]
        },
        {
            "article": "In a stunning upset, the Seoul Skyhawks defeated the Tokyo Thunder 89-87 in the final seconds of the Asia Pro Basketball League championship.",
            "label": ["Sports"]
        }
    ],
    'spell_check':[
            {
                "misspelled": "The quik brown fox jumps over the lazy dog.",  # 1
                "label": ["The quick brown fox jumps over the lazy dog.", "quick", "quik"]
            },
            {
                "misspelled": "She sells sea shals by the seashore.",  # 2
                "label": ["She sells sea shells by the seashore.", "shells", "shals"]
            },
            {
                "misspelled": "A journey of a thousand miles begin with a single step.",  # 3
                "label": ["A journey of a thousand miles begins with a single step.", "begins", "begin"]
            },
            {
                "misspelled": "To be or not to be, that is the qustion.",  # 4
                "label": ["To be or not to be, that is the question.", "question", "qustion"]
            },
            {
                "misspelled": "The only thing we have to fear is feer itself, except for spiders.",  # 5
                "label": ["The only thing we have to fear is fear itself, except for spiders.", "fear", "feer"]
            },
            {
                "misspelled": "The early bird catchs the worm, but the second mouse gets the cheese.",  # 6
                "label": ["The early bird catches the worm, but the second mouse gets the cheese.", "catches", "catchs"]
            },
            {
                "misspelled": "Actions speak loder than words, but silence speaks volumes.",  # 7
                "label": ["Actions speak louder than words, but silence speaks volumes.", "louder", "loder"]
            },
            {
                "misspelled": "A picture is worth a thousand words, but a good story is pricless.",  # 8
                "label": ["A picture is worth a thousand words, but a good story is priceless.", "priceless", "pricless"]
            },
            {
                "misspelled": "She is the best at what she do, and she does it with grace.",  # 9
                "label": ["She is the best at what she does, and she does it with grace.", "does", "do"]
            },
            {
                "misspelled": "The pen is mitier than the sword, but actions speak louder than both.",  # 10
                "label": ["The pen is mightier than the sword, but actions speak louder than both.", "mightier", "mitier"]
            }
    ],
    "adding_odd_numbers": [
        {
            "sequence": [2],    # 1
            "label": [0]
        },
        {
            "sequence": [1, 2, 3, 4, 5],    # 2
            "label": [9]
        },
        {
            "sequence": [1, 3, 5, 7, 9],    # 3
            "label": [25]
        },
        {
            "sequence": [10, 20, 30],   # 4
            "label": [0]
        },
        {
            "sequence": [111, 222, 333, 444, 555],   # 5
            "label": [999]
        },
        {
            "sequence": [100, 99],   # 6
            "label": [99]
        },
        {
            "sequence": [1, 2, 4, 6],   # 7
            "label": [1]
        },
        {
            "sequence": [11, 21, 31, 40],   # 8
            "label": [63]
        },
        {
            "sequence": [10, 20, 30, 41],   # 9
            "label": [41]
        },
        {
            "sequence": [2, 4, 6, 8],   # 10
            "label": [0]
        }
    ],
    "github_typo_check": [
        {
            "misspelled": "Follow the intructions carefully to complete the setup.",  # 1
            "label": ["Follow the instructions carefully to complete the setup.", "instructions", "intructions"]
        },
        {
            "misspelled": "Intial commit with basic project structure.",    # 2
            "label": ["Initial commit with basic project structure.", "Initial", "Intial"]
        },
        {
            "misspelled": "Plase make sure you have installed all dependencies.",   # 3
            "label": ["Please make sure you have installed all dependencies.", "Please", "Plase"]
        },
        {
            "misspelled": "The function retruns the result as a JSON object.",  # 4
            "label": ["The function returns the result as a JSON object.", "returns", "retruns"]
        },
        {
            "misspelled": "Set the enviroment variable before starting the server.",    # 5
            "label": ["Set the environment variable before starting the server.", "environment", "enviroment"]
        },
        {
            "misspelled": "This methd is deprecated and will be removed in future versions.",   # 6
            "label": ["This method is deprecated and will be removed in future versions.", "method", "methd"]
        },
        {
            "misspelled": "Don't foget to star the repository if you find it useful!",  # 7
            "label": ["Don't forget to star the repository if you find it useful!", "forget", "foget"]
        },
        {
            "misspelled": "Run `npm intall` to get started.",   # 8
            "label": ["Run `npm install` to get started.", "install", "intall"]
        },
        {
            "misspelled": "Fixed a bug in the `calcualteSum()` util.",  # 9
            "label": ["Fixed a bug in the `calculateSum()` util.", "calculateSum", "calcualteSum"]
        },
        {
            "misspelled": "Use `--forse` to override the default behavior.",    # 10
            "label": ["Use `--force` to override the default behavior.", "--force", "--forse"]
        }      
    ],
    "json_repair": [
        {
            "invalid_json_str": "{\"name\": \"Alice\", \"age\": 30, \"city\": \"New York\"",    # 1: missing closing brace
            "label": [{
                "name": "Alice",
                "age": 30,
                "city": "New York"
            }]
        },
        {
            "invalid_json_str": "{name: \"Bob\", age: 25, city: \"Chicago\"}",  # 2: missing quotes around keys
            "label": [{
                "name": "Bob",
                "age": 25,
                "city": "Chicago"
            }]
        },
        {
            "invalid_json_str": "{\"users\": {\"admin\", \"guest\", \"moderator\"}}",    # 3: curly braces instead of square brackets for array
            "label": [{
                "users": ["admin", "guest", "moderator"]
            }]
        },
        {
            "invalid_json_str": "{\"count\": 10,, \"status\": \"ok\"}", # 4: extra comma
            "label": [{
                "count": 10,
                "status": "ok"
            }]
        },
        {
            "invalid_json_str": "{\"enabled\": true \"thresholds\": 0.8}",   # 5: missing comma between key-value pairs
            "label": [{
                "enabled": True,
                "thresholds": 0.8
            }]
        },
        {
            "invalid_json_str": "{\"items\": [1, 2, 3], \"meta\": {\"total\": 3, \"page\": 1}", # 6: missing closing brace in nested object
            "label": [{
                "items": [1, 2, 3],
                "meta": {
                    "total": 3,
                    "page": 1
                }
            }]
        },
        {
            "invalid_json_str": "{\"config\": [\"path\": \"/usr/bin\", \"recursive\": true]}",  # 7: square brackets instead of curly braces for object
            "label": [{
                "config": {
                    "path": "/usr/bin",
                    "recursive": True
                }
            }]
        },
        {
            "invalid_json_str": "{\"success\": true, \"data\" null}", # 8: missing colon after "data"
            "label": [{
                "success": True,
                "data": None
            }]
        },
        {
            "invalid_json_str": "{'error': 'Not found', 'code': 404}",  # 9: single quotes instead of double quotes
            "label": [{
                "error": "Not found",
                "code": 404
            }]
        },
        {
            "invalid_json_str": "{\"features\": [\"a\", \"b\", \"c\"]]}",  # 10: extra closing bracket
            "label": [{
                "features": ["a", "b", "c"]
            }]
        }
    ],
    "model_name_extraction": [
        {
            "abstract": "The new model, GPT-4, has shown remarkable performance in various tasks.", # 1
            "label": [[["GPT-4"]]]
        },
        {
            "abstract": "In this paper, we introduce the BERT-Graph model.",  # 2
            "label": [[["BERT-Graph"]]]
        },
        {
            "abstract": "The latest version of the model, B-T5 (Bayesian T5), achieves state-of-the-art results in text generation tasks.", # 3
            "label": [[["B-T5", "Bayesian T5"]]]
        },
        {
            "abstract": "This paper investigates the combination of BalanceNet and ResNet for image classification tasks.", # 4
            "label": [[["BalanceNet"], ["ResNet"]]] 
        },
        {
            "abstract": "This study explores the use of diverse loss functions in deep learning.",    # 5
            "label": [[[]]]
        },
        {
            "abstract": "We propose a novel architecture of a code-specific deep learning model named CodeGenMix, specifically designed for code generation tasks.",   # 6
            "label": [[["CodeGenMix"]]]
        },
        {
            "abstract": "Our approach builds on top of RoBERTa and XLNet, adapting them for multilingual QA settings.", # 7
            "label": [[["RoBERTa"], ["XLNet"]]]
        },
        {
            "abstract": "Unlike prior work, our method does not rely on GPT-J or any other large-scale decoder-only model.",    # 8
            "label": [[["GPT-J"]]]
        },
        {
            "abstract": "The experiments were conducted using a modified version of VGG-19 and a lightweight variant of MobileNet.",    # 9
            "label": [[["VGG-19"], ["MobileNet"]]]
        },
        {
            "abstract": "We did not rely on any specific model, instead focusing on the overall architecture and training scheme.",   # 10
            "label": [[[]]]
        }
    ],
    "pos_detection": [
        {
            "sentence": "The quick brown fox jumps over the lazy dog.",  # 1
            "token": ["quick"],
            "label": ["JJ"]
        },
        {
            "sentence": "The quick brown fox jumps over the lazy dog.",  # 2
            "token": ["fox"],
            "label": ["NN"]
        },
        {
            "sentence": "The quick brown fox jumps over the lazy dog.", # 3
            "token": ["over"],
            "label": ["IN"]
        },
        {
            "sentence": "Those ideas are newer than the previous ones.",    # 4
            "token": ["newer"],
            "label": ["JJR"]
        },
        {
            "sentence": "Wow, that was an amazing performance!",    # 5
            "token": ["Wow"],
            "label": ["UH"]
        },
        {
            "sentence": "They have been working diligently all week.",  # 6
            "token": ["diligently"],
            "label": ["RB"]
        },
        {
            "sentence": "We visited Paris in 2022.",    # 7
            "token": ["2022"],
            "label": ["CD"]
        },
        {
            "sentence": "I like apples and oranges.",   # 8
            "token": ["and"],
            "label": ["CC"]
        },
        {
            "sentence": "The baseball player named John hit a home run.",  # 9
            "token": ["John"],
            "label": ["NNP"]
        },
        {
            "sentence": "This package includes everything you need.",   # 10
            "token": ["This"],
            "label": ["DT"]
        }
    ]
}


if __name__ == "__main__":
    model = 'llama'
    
    tasks = ['syntactic_bug_detection', 'topic_classification', 'spell_check', 'adding_odd_numbers', 'github_typo_check', 'json_repair', 'model_name_extraction', 'pos_detection']
    produce_inference_results = True
    
    if model == 'llama':
        target_layers = [21]
        
    elif model == 'mistral':
        target_layers = [22]
        
    elif model == 'gemma':
        target_layers = [28]

    for task in tqdm(tasks):
        print(f"Processing task: {task}")
        LIH_init = defaultdict(list)
        test_scores_init = []
        labels_init = []

        prompt_template_name = 'messages_template'

        for i, test_case in enumerate(initial_tests[task]):
            inference_outputs, layer2hidden = inference.from_input(
                input_variables=test_case,
                task=task,
                prompt_template_name=prompt_template_name,
                num_inference_runs=10,
                target_layers=target_layers,
                produce_inference_results=produce_inference_results,
                model=model
            )
            
            for l in layer2hidden:
                LIH_init[l].append(layer2hidden[l])
                
            if produce_inference_results:
                _test_score, _labels = inference.evaluate(inference_outputs, test_case['label'], task=task, prompt_template_name=prompt_template_name)

                test_scores_init.append(_test_score)
                labels_init.append(_labels)

                if _test_score == 0:
                    print(inference_outputs[0])
                print(f"Test Case {i+1}: score {_test_score}, labels {_labels}", end='\n---\n')


        os.makedirs(f'./initial_tests_{model}/{task}', exist_ok=True)

        for l, hidden_vectors in LIH_init.items():
            hidden_vectors = np.array(hidden_vectors)
            print(f"Layer {l} hidden vectors shape: {hidden_vectors.shape}")
            torch.save(hidden_vectors, f'./initial_tests_{model}/{task}/hidden_vectors_layer_{l}.pt')

        if produce_inference_results:
            with open(f'./initial_tests_{model}/{task}/inference_results.json', 'w') as f:
                json.dump({
                    'test_scores': test_scores_init,
                    'test_labels': labels_init
                }, f, indent=2)