from datasets import load_dataset, Dataset
import pandas as pd
from langdetect import detect
from tqdm import tqdm
import re
import pickle
from typing import List, Dict, Union
default_instruction_prompt = '''A helpful assistant, who is an expert in the fields of philosophy and literary studies, takes questions and tasks from a human. The assistant provides responses that appropriately complete the request in the same language that the human used. \n'''


def get_languages(ds: Dataset, langs: List[str] =['de','en'], output: str ='dataset') -> Union[pd.DataFrame, Dict[str, List[str]]]:
    """
    Filter out the dataset based on given languages and return a dictionary of texts or pandas DataFrame.

    Parameters:
    ds (Dataset): HuggingFace Dataset object.
    langs (List[str]): List of language codes to filter the dataset.
    output (str): Output format - 'dataset' or 'text'.

    Returns:
    Union[pd.DataFrame, Dict[str, List[str]]]: Pandas DataFrame or dictionary containing texts based on the output format.
    """
    inst_res_dict = {'lang':[],'instruction':[],'response':[]}
    text_dict = {'text':[]}

    for sample in tqdm(ds['train']['text']+ds['test']['text']):
        sample_prep = re.sub(r'### Human:',' ',sample)
        sample_prep = sample_prep.split('### Assistant:')
        try:                                            
            lang = detect(sample_prep[0][:150])

            if lang in langs: 
                if output == 'dataset':
                    inst_res_dict['lang'].append(lang)
                    inst_res_dict['instruction'].append(sample_prep[0].strip())
                    inst_res_dict['response'].append(sample_prep[1].strip())
                elif output == 'text':
                    text = default_instruction_prompt + '\n'
                    text += sample
                    text_dict['text'].append(text)
        except:
            pass

    if output == 'dataset':
        return pd.DataFrame.from_dict(inst_res_dict)
    elif output == 'text':
        return text_dict
    

if __name__ == "__main__":
    num_samples_per_lang = 6000
    dataset = load_dataset("timdettmers/openassistant-guanaco")
    save_to = 'data/datasets/'
    filename = 'DettmersGuanaco_complete'

    langs = ['de','en']
    output = 'text'

    dataset_df = get_languages(dataset,langs=langs,output=output)

    if output == 'dataset':
        dataset_export_df = pd.DataFrame.from_dict({'lang':[],'instruction':[],'response':[]})
        for lang in langs: 
            lang_df = dataset_df[dataset_df['lang']==lang].sample(frac=1)[:num_samples_per_lang]

            dataset_export_df = pd.concat([dataset_export_df,lang_df])
        print('length of exported datset')
        print(len(dataset_export_df))
        print(dataset_export_df.sample(frac=1).head())

        dataset_df.to_pickle(save_to+filename+'.pkl')
    elif output == 'text':
        train_folder = 'data/dataset_processed/train/'
        eval_folder = 'data/dataset_processed/eval/'
        ds = Dataset.from_dict(dataset_df).shuffle()
        ds = ds.train_test_split(test_size = 0.1)

        print(ds)

        print(ds['test'][300])
        print(ds['train'][100])
        with open(train_folder+filename+'.pkl', 'wb') as handle:
            pickle.dump(ds['train'], handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(eval_folder+filename+'.pkl', 'wb') as handle:
            pickle.dump(ds['test'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f'{output} is not an implemented output format')
