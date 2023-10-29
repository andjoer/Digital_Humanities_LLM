"""
Program Description: 

This program is a script for data preparation for training large language models. The script works on an annotated dataset in which all tokens are labeled. It performs several tasks:

1. The script reads data from multiple data paths. The paths are of English and German annotated datasets stored in '.txt' format. 

2. It prepares instructions in both English and German languages to be provided to the language model. The instructions request extraction of different components of the arguments presented in the text.

3. The script then defines multiple utility functions. These include transforming dictionaries into responses, converting text into tasks, creating label tasks, separating lines into tokens and labels, preparing text by arranging it along with labels, processing lines to return texts and labels, and other related tasks.

4. The main part of the script reads the '.txt' files from the provided paths, processes the lines, and prepares texts and dictionaries of labels. It also decides the language of the instruction and response (either English or German) and prepares the tasks.

5. Finally, the script constructs a Pandas DataFrame containing the instructions and responses, along with the language, and saves it as a Pickle file. It also writes all the instructions and responses in a '.txt' file for reference.
"""


import pandas as pd
import numpy as np

import random


from typing import Dict, List, Tuple, Union

path_de_arg: List[str]  = ['data/external_datasets/UKP_arg_mining/Orig/train_MTX.de','data/external_datasets/UKP_arg_mining/HT/train_PE.de']
path_en_arg: List[str]  = ['data/external_datasets/UKP_multi_arg_mining/fullData/web/train.txt','data/external_datasets/UKP_multi_arg_mining/fullData/wiki/train.txt']


prompt_de = '''Im folgenden Textausschnitt kommen Argumente vor. Diese lassen sich sprachlich in Hauptthese (Major-Claim), These (Claim), Unterstützung (Backing), Prämisse (Premise) und Beweis (Evidence) unterteilen. Bitte extrahiere die einzelnen Komponenten der Argumente. Sollte ein Abschnitt unter eine dieser Kategorien fallen, schreibe ihn bitte unter 
die entsprechende Überschrift. Eine Ausgabe für die Aufgabe könnte so aussehen: 
Claim:
"Claim 1"
"Claim 2"
...
Premise:
"Premise 1"
...
...

Dies ist der zu bearbeitende Textausschnitt: \n'''

response_de = '''Dies sind die argumentativ relevanten Textstellen, einsortiert unter die jeweilige Kategorie:'''


prompt_en = '''In the following text excerpt, arguments are present. These can be linguistically categorized into major claim, claim, backing, premise, and evidence. Please extract the different components of the arguments.
If a paragraph falls under any of these categories, please write it under the corresponding heading. And example output could look like this: : 
Claim:
"Claim 1"
"Claim 2"
...
Premise:
"Premise 1"
...
...

This is the text excerpt to be analyzed: \n'''

response_en = '''These are the argumentativeley relevant passages, sorted into their respective categories:'''

prompt_de_label = '''Im folgenden Textausschnitt kommen Argumente vor. Diese lassen sich sprachlich in Hauptthese (Major-Claim), These (Claim), Unterstützung (Backing), Prämisse (Premise) und Beweis (Evidence) unterteilen. 
Bitte annotiere die Komponenten der Argumente. 

Im folgenden ist eine durchnummerierte Liste von zu annotierenden Sätzen. Bitte erstelle eine Liste mit diesen Nummern und den entsprechenden Labels. Sollten die Sätze argumentativ nicht relevant sein ist das Label "O".

{sents}

Dies ist der Kontext der Sätze: \n {context}'''

response_de_label = '''Dies ist die Liste mit den entsprechenden Labels:'''


prompt_en_label = '''In the following text excerpt, arguments occur. These can be linguistically divided into major claim, claim, backing, premise, and evidence.
Please annotate the components of the arguments.

This is a list of the sentences that should be annotated. Please create a list with these numbers and the corresponding labels. If the sentences are not argumentatively relevant, the label is 'O'.

{sents}

This is the context of the sentences: \n {context}'''

response_en_label = '''This is the list with the corresponding labels:'''

def dict_to_response(dct: Dict, response: str) -> str:
    """
    Converts a dictionary to response string

    Args:
        dct (Dict): The dictionary to be converted.
        response (str): The response string.

    Returns:
        str: The response string with the dictionary converted to string
    """
    for key in dct.keys():
        if key != 'O':
            response += '\n' + key +':\n'
            response += '\n'.join(dct[key])

    return response

def text_to_task(text: str, prompt: str) -> str:
    """
    Converts a text to task

    Args:
        text (str): The text to be converted.
        prompt (str): The prompt string.

    Returns:
        str: The converted task string
    """

    return prompt + '\n' + text



def create_label_task(text: str, dct: Dict, prompt: str, response: str, num_samples: int = 2, min_len: int = 4) -> Tuple[str, str]:
    """
    Creates a label task from a text

    Args:
        text (str): The text to be labeled.
        dct (Dict): The dictionary with labels.
        prompt (str): The prompt string.
        response (str): The response string.
        num_samples (int, optional): The number of samples. Defaults to 2.
        min_len (int, optional): The minimum length of a sample. Defaults to 4.

    Returns:
        Tuple[str, str]: The instruction and response as a tuple.
    """

    sents = ''
    annotation = ''

    chosen = [] 

    lens = []
    
    for key in dct.keys():

        if key != 'O':
            lens.append(max([len(item.split()) for item in dct[key]]))

        dct[key] = [item for item in dct[key] if len(item) > min_len]
        if dct[key]:
            sentence = random.sample(dct[key],min(num_samples,len(dct[key])))

            chosen += [(sample,key) for sample in sentence]

    random.shuffle(chosen)
    try:
        max_len = max(lens)
    except: 
        max_len = 20
    for num, sample in enumerate(chosen):
        sent = sample[0]

        if sample[1] == 'O':
            if len(sent.split()) > max_len + 4:
                rand = random.randint(0,1)                         #50% of the too long sequences will be shorteneds
                if rand == 1:
                    sent = sent.split('.')[0]

      
        sents += str(num) +') ' + sent + '\n\n'

        annotation += str(num) +') '+ ' Label: ' + sample[1] + '\n\n'


    instruction = prompt.format(sents=sents,context=text)

    response += '\n'+annotation

    return instruction,response


def sep_line(line: str) -> Tuple[str, str]:
    """
    Separates a line into token and label

    Args:
        line (str): The line to be separated.

    Returns:
        Tuple[str, str]: The separated token and label.
    """
    line_sp = line.split('\t')
    token = line_sp[0].split('_')[0]
    if len(token) > 0 and token[0].isalpha():
        token = ' ' + token
    try:
        label = line_sp[1].split('-')[1].strip()
    except:
        label = 'O'

    return (token, label)

def prepare_text(tokens: List[str], labels: List[str]) -> Tuple[str, Dict]:
    """
    Prepares a text from tokens and labels

    Args:
        tokens (List[str]): The list of tokens.
        labels (List[str]): The list of labels.

    Returns:
        Tuple[str, Dict]: The prepared text and a dictionary with labels.
    """
    text = ''.join(tokens).strip()

    state = 'O'
   
    labeled_dict = {}

    for label in set(labels):

        labeled_dict[label] = []

    labeled_text = []
    
    
    for idx, token in enumerate(tokens):
        if state != labels[idx] or idx == len(tokens)-1:

            txt = ''.join(labeled_text).strip()

            if txt:
                labeled_dict[state].append(txt)

        
            state = labels[idx]
  
            labeled_text = []

        labeled_text.append(token)

    return text, labeled_dict

def line_batches(tokens: List[str], sep: str = '\n') -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits tokens into batches by a separator

    Args:
        tokens (List[str]): The list of tokens.
        sep (str, optional): The separator. Defaults to '\n'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The indices of new lines and the distances between them.
    """
    new_lines = np.asarray([[i for i,val in enumerate(tokens) if val==sep]])[0,:]
    
    distances = np.ediff1d(new_lines)
    return new_lines, distances

def process_lines(lines: List[str], max_words: int = 600, max_max_words: int = 800, min_len: int = 10, max_too_long_ratio: float = 0.2) -> Tuple[List[str], List[Dict]]:
    """
    Processes lines to generate texts and labeled dictionaries

    Args:
        lines (List[str]): The lines to be processed.
        max_words (int, optional): The maximum number of words. Defaults to 600.
        max_max_words (int, optional): The maximum of maximum words. Defaults to 800.
        min_len (int, optional): The minimum length of a text. Defaults to 10.
        max_too_long_ratio (float, optional): The maximum ratio of too long sequences. Defaults to 0.2.

    Returns:
        Tuple[List[str], List[Dict]]: The list of texts and list of labeled dictionaries.
    """
    texts = []
    labeled_dicts = []
    tokens, labels = zip(*[sep_line(line) for line in lines])

    new_lines, distances = line_batches(tokens)

    too_long = len(distances[distances > max_max_words])

    if too_long > 0 and len(new_lines)/too_long > max_too_long_ratio:   # if there are too many too long sequences split the text by comma not by new line
        new_lines, _ = line_batches(tokens,sep='.')

    last_cut = 0
    while last_cut < len(tokens): 
        next_diff = new_lines[new_lines > last_cut]
        if len(next_diff)>0 and next_diff.min() - last_cut < max_max_words:
            optimum = last_cut + max_words

            for i in range(2):
            
                next_cut = new_lines[new_lines < optimum].max()
            

                if next_cut != last_cut: 
                    break
                else:
                    optimum = last_cut + max_max_words

            if next_cut == last_cut:
                next_cut = len(tokens)

            if next_cut - last_cut > min_len:
                text, labeled_dict = prepare_text(tokens[last_cut:next_cut],labels[last_cut:next_cut])
                texts.append(text)
                labeled_dicts.append(labeled_dict)

            last_cut = next_cut

        else:

            idx = np.where(new_lines == last_cut)[0][0]

            if idx < len(new_lines)-1:
                last_cut = new_lines[idx+1]
            else:
                break

    return texts, labeled_dicts


texts_de = []
labeled_dicts_de = []

files_de = []
for path in path_de_arg:
    with open(path) as f:
        lines = f.readlines()
    texts, labeled_dicts = process_lines(lines)
    texts_de += texts
    labeled_dicts_de += labeled_dicts
    files_de +=[path]*len(labeled_dicts)
    

texts_en = []
labeled_dicts_en = []

files_en = []
for path in path_en_arg:
    with open(path) as f:
        lines = f.readlines()

    texts, labeled_dicts = process_lines(lines)

    texts_en += texts
    labeled_dicts_en += labeled_dicts
    
    files_en +=[path]*len(labeled_dicts)
langs = []
instructions = []
responses = []

files = []
for idx, text in enumerate(texts_de):
    langs.append('de')
    
    rand = random.randint(0,2)

    if rand == 1:
        instructions.append(text_to_task(text,prompt_de))
        responses.append(dict_to_response(labeled_dicts_de[idx],response_de))
    else:
        instruction, response = create_label_task(text,labeled_dicts_de[idx],prompt_de_label,response_de_label,num_samples = 2)
        instructions.append(instruction)
        responses.append(response)

    files.append(files_de[idx])
    

for idx, text in enumerate(texts_en):
    langs.append('en')

    rand = random.randint(0,2)
    if rand == 1:
        instructions.append(text_to_task(text,prompt_en))
        responses.append(dict_to_response(labeled_dicts_en[idx],response_en))


    else:
        instruction, response = create_label_task(text,labeled_dicts_en[idx],prompt_en_label,response_en_label,num_samples = 2)
        instructions.append(instruction)
        responses.append(response)
    files.append(files_en[idx])

data = {'instruction': instructions, 'response': responses,'lang':langs}
train_df = pd.DataFrame(data)

train_df.to_pickle('data/datasets_to_process/arguments.pkl')

export_text = ''

for idx, instruction in enumerate(instructions):
    export_text += files[idx] +'\n'
    export_text += '\n instruction: ' + instruction
    export_text += '\n\n response: ' + responses[idx] +'\n\n'

with open('arguments_check.txt', 'w') as f:
        
    f.write(export_text)