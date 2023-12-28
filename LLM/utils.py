import re
from openai import OpenAI, AsyncOpenAI
import pandas as pd
from collections import defaultdict
import time
import os
import json

import aiohttp
import asyncio
import os

def formatting_func_standard(ds, eos_token=None):

    if eos_token is None:                              # in evaluation step
          eos_token = ''
    ds_lst = ds['text']
    for idx, string in enumerate(ds_lst):
        formatted_text = re.sub('### Assistant','### assistant',string) + eos_token
        ds_lst[idx] = formatted_text

    return {'text':ds_lst}

def formatting_func_json_gpt3(json_data_list):
    print('formatting json for gpt3.5')

    ds_lst = []

    for json_data in json_data_list:

        for message in json_data['messages']:
            if message['role'] == 'user':
                formatted_output = "### human: " + message['content'] + "\n"
            elif message['role'] == 'assistant':
                formatted_output += "### assistant: " + message['content'] + "\n"

        ds_lst.append(formatted_output)

    return {'text':ds_lst}

def formatting_func_dataframe(df):

    ds_lst = []

    for index, row in df.iterrows():
        formatted_output = "### human: " + row['instruction'] + "\n"
        formatted_output += "### assistant: " + row['response'] + "\n"

        ds_lst.append(formatted_output)

    return {'text': ds_lst}



def formatting_func_api(ds, eos_token=None):

    if eos_token is None:                              # in evaluation step
          eos_token = ''
    ds_lst = ds['text']
    for idx, string in enumerate(ds_lst):
        formatted_text = re.sub('### Assistant','### assistant',string)
        formatted_text = re.sub('### Human','### human',string)
        formatted_text = re.sub('### assistant','',string) 
        formatted_text = re.sub('### human','',string) 
        ds_lst[idx] = formatted_text

    return {'text':ds_lst}

def formatting_func_chat(ds):
            
            ds_lst = ds['text']
            for idx, string in enumerate(ds_lst):
                formatted_text = re.sub('### Assistant:','[/INST]',string)
                formatted_text = re.sub('### assistant:','[/INST]',formatted_text)
                formatted_text = re.sub('### Human','### human',formatted_text)
                formatted_text = '[/INST]'.join(formatted_text.split('### human:')[1:])
                system_prompt = '''A helpful assistant, who is an expert in the fields of philosophy and literary studies, takes questions and tasks from a human. The assistant provides responses that appropriately complete the request in the same language that the human used.'''
                chat_text = f'''[INST] <<SYS>>{
                    system_prompt}
                    <</SYS>> 
                    {formatted_text}</s>'''
                ds_lst[idx] = chat_text

            return {'text':ds_lst}

def formatting_gpt4_anno(df):

    print('apply custom formatting for gpt4 annotation')
    file_path = './prompts/prompt_annotation.txt'
    # Reading the content of the text file
    with open(file_path, 'r') as file:
        prompt_annotation = file.read().strip()

    # Creating a list to store the formatted strings
    formatted_strings = []

    # Iterating through each row of the DataFrame
    for index, row in df.iterrows():
        # Formatting the string as specified
        formatted_string = f"### human: {prompt_annotation} \n {row['examp_excerpts']} \n ### assistant: {row['response']}"
        formatted_strings.append(formatted_string)

    return {'text':formatted_strings}

def formatting_gpt3_anno(df):

    print('apply custom formatting for gpt3 annotation')
    file_path = './prompts/prompt_annotation_gpt3.txt'
    # Reading the content of the text file
    with open(file_path, 'r') as file:
        prompt_annotation = file.read().strip()

    # Creating a list to store the formatted strings
    formatted_strings = []

    # Iterating through each row of the DataFrame
    for index, row in df.iterrows():
        # Formatting the string as specified
        formatted_string = f"### human: {prompt_annotation} \n\n {row['examp_excerpts']} \n ### assistant: {row['response']}"
        formatted_strings.append(formatted_string)

    return {'text':formatted_strings}

def correct_annotations(data_dict,wiedergabe = None):
    corrected_data = {}
    figurenrede = False
    author = False
    if wiedergabe is not None: 
        if "erzählstimme" in wiedergabe.lower():
            author = True

    for key, value in data_dict.items():
        key_lower = key.lower()
        value_str = str(value)  # Convert value to string for text processing

        
        if 'wiedergabeform' in key_lower and 'erzählstimme' in value_str.lower() and ('indirekt' in value_str.lower() or 'direkt' in value_str.lower()):
            value_str = value_str.replace('erzählstimme', 'figurenrede')
            value_str = value_str.replace('Erzählstimme', 'figurenrede')
            
        if 'wiedergabeform' in key_lower and ('direkt' in value_str.lower() or 'indirekt' in value_str.lower()): 
            if 'figurenrede' not in value_str.lower():
                value_str = value_str.replace('wiedergabeform', 'figurenrede, wiedergabeform')
                value_str = value_str.replace('Wiedergabeform', 'Figurenrede, Wiedergabeform')
            figurenrede = True
      

        if 'in_gedankengang' in key_lower and figurenrede and not author:
    
            value_str = value_str.replace('ja', 'nein')
            value_str = value_str.replace('Ja', 'Nein')

        corrected_data[key] = value_str

    return corrected_data

async def fetch(client, prompt, model):
    try:
        response = await client.chat.completions.create(
            model=model,       
            temperature=0,
            messages=[
                {"role": "system", "content": "Du bist der führende deutsche Literaturwissenschaftler Prof. Dr. Andreas Huysse. Deine Tätigkeit ist in der Quantitativen Literaturwissenschaft. Deine Aufgabe ist es Textstellen zu annotieren. Du argumentierst ausführlich und stets mit Pro und Kontra. Deine Texte werden lieber zu lang als zu kurz."},
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print('Failed to connect with API:', e)
        return ''

async def fetch_all(prompts, model):
    async with AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY")) as client:
        tasks = [fetch(client, prompt, model) for prompt in prompts]
        return await asyncio.gather(*tasks)


def openAi_gen_batch(prompts,model):
    print('processing batch with length ' + str(len(prompts)))
    return asyncio.run(fetch_all(prompts, model))


def openAi_gen(prompt,model):

    if 'gpt-3' in model:
        system_message ='Du bist der führende deutsche Literaturwissenschaftler Prof. Dr. Andreas Huysse. Deine Tätigkeit ist in der Quantitativen Literaturwissenschaft. Deine Aufgabe ist es Textstellen zu annotieren.'

    #system_message = "Du bist der führende deutsche Literaturwissenschaftler Prof. Dr. Andreas Huysse. Deine Tätigkeit ist in der Quantitativen Literaturwissenschaft. Deine Aufgabe ist es Textstellen zu annotieren. Du argumentierst ausführlich und stets mit Pro und Kontra. Deine Texte werden lieber zu lang als zu kurz."
   
    for i in range (20):

        try:
            client = OpenAI(
                # This is the default and can be omitted
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
            response = client.chat.completions.create(
                
                    model=model,       
                    temperature = 0,
                    messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt},
                        ]
                    )
            
        except: 
            print('failed to connect with api')
            time.sleep(30)

    return response.choices[0].message.content

def calculate_confusion_matrix(tuples):
    """
    Calculate the confusion matrix as proportions from a list of tuples (prediction, ground_truth).

    :param tuples: List of tuples (prediction, ground_truth)
    :return: Pandas DataFrame representing the confusion matrix as proportions
    """
    matrix = defaultdict(lambda: defaultdict(int))

    # Count occurrences of each (prediction, ground_truth) pair
    for prediction, ground_truth in tuples:
        matrix[prediction][ground_truth] += 1

    # Convert to DataFrame for better visualization
    df_matrix = pd.DataFrame(matrix).fillna(0).astype(int)

    # Ensure all predictions and ground truths are represented
    all_labels = set([pred for pred, _ in tuples]) | set([gt for _, gt in tuples])
    df_matrix = df_matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)

    return df_matrix

def calculate_f1_scores(confusion_matrix):
    """
    Calculate F1 scores for each category in the confusion matrix.

    :param confusion_matrix: Pandas DataFrame representing the confusion matrix
    :return: Dictionary with categories as keys and their F1 scores as values
    """
    f1_scores = {}

    for category in confusion_matrix.columns:
        # True Positives (TP)
        tp = confusion_matrix.loc[category, category]

        # False Positives (FP) - Sum of column for category minus TP
        fp = confusion_matrix[category].sum() - tp

        # False Negatives (FN) - Sum of row for category minus TP
        fn = confusion_matrix.loc[category].sum() - tp

        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores[category] = f1

    return f1_scores
#print(calculate_confusion_matrix([('b','c'),('b','b'),('b','a')]))