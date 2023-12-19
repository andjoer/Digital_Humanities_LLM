import re
from openai import OpenAI
import pandas as pd
from collections import defaultdict
import time
import os

def formatting_func_standard(ds, eos_token=None):

    if eos_token is None:                              # in evaluation step
          eos_token = ''
    ds_lst = ds['text']
    for idx, string in enumerate(ds_lst):
        formatted_text = re.sub('### Assistant','### assistant',string) + eos_token
        ds_lst[idx] = formatted_text

    return {'text':ds_lst}


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


def openAi_gen(prompt):
   
    #for i in range (20):

        #try:

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    response = client.chat.completions.create(
            model="gpt-4-1106-preview",       
            temperature = 0,
            messages=[
                    {"role": "system", "content": "Du bist der führende deutsche Literaturwissenschaftler Prof. Dr. Andreas Huysse. Deine Tätigkeit ist in der Quantitativen Literaturwissenschaft. Deine Aufgabe ist es Textstellen zu annotieren."},
                    {"role": "user", "content": prompt},
                ]
            )
        #    break
            
        #except: 
         #   print('failed to connect with api')
          #  time.sleep(10)
    print(response)
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

    # Normalize the matrix to show proportions
    total_instances = sum(sum(row.values()) for row in matrix.values())
    if total_instances > 0:
        df_matrix = df_matrix #/ total_instances

    # Ensure all predictions and ground truths are represented
    all_labels = set([pred for pred, _ in tuples]) | set([gt for _, gt in tuples])
    df_matrix = df_matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)

    return df_matrix