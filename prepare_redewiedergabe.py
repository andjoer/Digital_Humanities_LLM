"""
This Python program is designed to process a labeled collection of text data for 
speech, thought, and writing representation (STWR). It first prepares labels for each 
text in the dataset. Tokens are concatenated into a single string and are then used to generate instructions 
and responses. The program finally compiles all the responses and instructions, processes 
each file, and stores the processed data in a dataframe, which is then saved as a pickle file. 
A text file is also created containing all the files chosen, the instructions, and the responses.
"""

import pandas as pd
import glob
import random
from collections import Counter
from typing import Tuple, List

path = 'data/external_datasets/redewiedergabe/*.tsv'

files = glob.glob(path)
labels: List[str] = []


def prepare_label(stwr: str) -> Tuple[str, str, str, str]:
    """
    Prepares the labels for each text in the dataset.

    Args:
    stwr: The text string to be labeled.

    Returns:
    A tuple with the labels.
    """
    if stwr == '-':
        return '-', '0', '-', '0'

    stwr_split = stwr.split('|')

    if len(stwr_split) > 1:
        sub_split = stwr_split[1].split('.')
        sub = sub_split[0] + ' ' + sub_split[1]
        sub_id = sub_split[2]
    else:
        sub = '-'
        sub_id = '0'

    main_split = stwr_split[0].split('.')
    main = main_split[0] + ' ' + main_split[1]
    main_id = main_split[2]
    return main, main_id, sub, sub_id


def get_response_label(corpus_df: pd.DataFrame, num_stwr_sent: int = 3, num_no_stwr_sent: int = 3, min_words: int = 4, max_len_factor: int = 2) -> pd.DataFrame:
    """
    Get the labels for the responses in the dataframe.

    Args:
    corpus_df: The dataframe containing the responses.
    num_stwr_sent: Number of sentences in STWR.
    num_no_stwr_sent: Number of sentences not in STWR.
    min_words: Minimum number of words in a sentence.
    max_len_factor: The factor by which the maximum length can exceed the minimum length.

    Returns:
    A dataframe with the labeled responses.
    """

    token_dict = {'tok': [], 'main': [], 'main_id': [], 'sub': [], 'sub_id': [], 'id': []}
    state_main_id = 0
    state_sub_id = 0
    id = 0
    state = '-'
    for _, row in corpus_df.iterrows():
        main, main_id, sub, sub_id = prepare_label(row['stwr'])
        token_dict['tok'].append(row['tok'])
        token_dict['main'].append(main)
        token_dict['main_id'].append(int(main_id))
        token_dict['sub'].append(sub)
        token_dict['sub_id'].append(int(sub_id))

        if main_id != state_main_id or sub_id != state_sub_id or (main == '-' and state != '-'):
            id += 1

        token_dict['id'].append(id)
        state = main
        state_main_id = main_id
        state_sub_id = sub_id

    id_counts = dict(Counter(token_dict['id']))


    token_df = pd.DataFrame.from_dict(token_dict)

    token_df['len'] = token_df['id'].apply(lambda x: id_counts[x])

    token_df = token_df[token_df['len'] >= min_words].reset_index(drop=True)

    token_df_stwr = token_df[token_df['main'] != '-']
    token_df_no_stwr = token_df[token_df['main'] == '-']

    stwr_idxs = list(set(list(token_df_stwr['id'])))
    stwr_ids = random.sample(stwr_idxs,min(num_stwr_sent,len(stwr_idxs)))

    no_stwr_idxs = list(set(list(token_df_no_stwr['id'])))
    no_stwr_ids = random.sample(no_stwr_idxs,min(num_no_stwr_sent,len(no_stwr_idxs)))

    sent_dict = {'text':[],'label':[],'id':[],'len':[]}

    for id in stwr_ids:

        text_df = token_df_stwr[token_df_stwr['id'] == id].reset_index()

        sent_dict['text'].append(get_text_body(list(text_df['tok'])))

        label_text = text_df.loc[0,'main']

        if text_df.loc[0,'sub'] != '-':
            label_text += ', ' + text_df.loc[0,'sub']

        sent_dict['label'].append(label_text)

        global labels

        labels.append(label_text)

        sent_dict['id'].append(id)

        sent_dict['len'].append(len(text_df))

    try:
        max_len = max(sent_dict['len'])
    except:
        max_len = 25

    for id in no_stwr_ids:
        text_df = token_df_no_stwr[token_df_no_stwr['id'] == id].reset_index()

        ## randomly shorten long sequencies so does it will not learn that long sequences have no stwr tag
        if len(text_df) > max_len:
            rand = random.randint(0,1)

            if rand == 1 or len(text_df) > max_len*max_len_factor:

                rand = random.randint(0,1)

                if rand == 1:
                    max_idx = max_len + random.randint(-3,3)
                    text_df = text_df[:max_idx] 
                else:
                    try:
                        fullstops = [index for (index, item) in enumerate(text_df['tok'][:-(min_words+4)]) if '.' in item]
                    except:
                        fullstops = []
                    if fullstops:
                        fullstop = random.randint(0,len(fullstops)-1)

                        if fullstop == len(fullstops) -1:
                            text_df = text_df[fullstops[fullstop]:]
                        else:
                            text_df =  text_df[fullstops[fullstop]:fullstops[fullstop]+1]
                    else:
                        max_idx = max_len + random.randint(-3,3)
                        text_df = text_df[:max_idx] 

        #####

        sent_dict['text'].append(get_text_body(list(text_df['tok'])))
        sent_dict['label'].append('Keine Redewiedergabe')
        sent_dict['id'].append(id)
        sent_dict['len'].append(len(text_df))

    return pd.DataFrame.from_dict(sent_dict).sort_values(by=['id']).reset_index(drop=True)


def get_text_body(tokens: List[str]) -> str:
    """
    Concatenate tokens into a single string, adding spaces before each token
    except those that start with comma or dot.
    
    Args:
        tokens (List[str]): List of tokens to be concatenated.
        
    Returns:
        str: Concatenated text.
    """
    text = ''
    for token in tokens:
        token = str(token)
        if token[0] not in [',','.']:
            token = ' ' + token
        text += token
    return text

def get_instruction_response(corpus_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Generate instructions and responses based on a corpus DataFrame.
    
    Args:
        corpus_df (pd.DataFrame): Dataframe with corpus data.
        
    Returns:
        Tuple[str, str]: Instructions and responses in a tuple.
    """
    try:

        tokens = list(corpus_df['tok'])
    except: 
        return '', ''

    labeled_sents = get_response_label(corpus_df)

    sents = ''
    annotation = ''

    for num, row in labeled_sents.iterrows():
        sents += str(num)+') '+row['text'] + '\n\n'
        annotation += str(num)+') ' + ' Label: ' +row['label'] + '\n\n' #row['text'] + ' Label: ' +row['label'] + '\n'

    context = get_text_body(tokens)

    instruction= f'''Der folgende Text soll im Hinblick auf die Redewiedergabe annotiert werden. Die Labels sind in englischer Sprache.   
Die Haupt-Annotation für Wiedergaben erfolgt mit dem Tag 'STWR' (speech, thought, writing representation).
Dessen wichtigste zwei Attribute (medium und type) kategorisieren die Wiedergaben auf zwei Achsen:
1) medium 	Was wird wiedergegeben? speech (Rede), thought (Gedanken), writing (Geschriebenes)
2) type 	Auf welche Weise wird wiedergegeben? 	 direct (direkt) z.B. "Er sagte: "Ich bin hungrig."",
free indirect (frei-indirekte Rede, 'erlebt') z.B. "Wo sollte er nur jetzt etwas zu Essen herbekommen?"
indirect (indirekte Rede) z.B. "Er sagte, er sei hungrig."
reported (erzählt) 	z.B. "Sie sprachen über Restaurants."
Sonderfall: indirect/free indirect ('berichtete Rede') 	z.B. "Wo sei das beste Restaurant zu finden?"

Es kann sein, dass ein Satz mehrere Labels hat.

Bitte annotiere folgende nummerierte Sätze (zähle die Nummern auf und schreibe das Label dahinter) 

{sents}

Dies ist der Kontext: \n {context}'''  #Quelle: http://redewiedergabe.de/richtlinien/richtlinien.html

    
    
    response = f'''Dies ist eine Liste mit den entsprechenden Labels:
    {annotation}'''
    return instruction, response

responses = []
instructions = []

files_chosen = []
for file in files: 
    print(file)
    try:
        corpus_df = pd.read_table(file)
        process = True
        
    except: 
        print('error with file' + str(file))
        process = False

    if process:
        instruction, response = get_instruction_response(corpus_df)

        if instruction and response and('\t-\t-' not in instruction+response):  # quick fix for parsing errors

            responses.append(response)
            instructions.append(instruction)
            files_chosen.append(file)


data = {'instruction': instructions, 'response': responses}
train_df = pd.DataFrame(data)
train_df.to_pickle('data/datasets_to_process/redewiedergabe.pkl')

print(set(labels))

export_text = ''

for idx, instruction in enumerate(instructions):
    export_text += files_chosen[idx] +'\n'
    export_text += '\n instruction: ' + instruction
    export_text += '\n\n response: ' + responses[idx] +'\n\n'

with open('redewiedergabe_check.txt', 'w') as f:
        
    f.write(export_text)