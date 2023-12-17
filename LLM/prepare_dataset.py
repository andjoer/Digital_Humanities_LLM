import pandas as pd

import pickle

import random

from typing import List, Tuple, Dict, Union
import os
from dataclasses import dataclass, field
from typing import Optional

import glob
from datasets import Dataset, concatenate_datasets, DatasetDict


from transformers import HfArgumentParser

default_instruction_prompt = '''A helpful assistant, who is an expert in the fields of philosophy and literary studies, takes questions and tasks from a human. The assistant provides responses that appropriately complete the request in the same language that the human used. \n'''

from typing import List, Tuple, Dict, Union
from datasets import Dataset
import pandas as pd

class PrepareDataset:

    """
    A class used to prepare a dataset for some type of text processing.

    ...

    Attributes
    ----------
    input_dir : str
        a formatted string to define the input directory (default "./data/datasets")
    instruction_prompt : str
        a formatted string to define the system prompt (default is a predefined string)
    format_human : str
        a formatted string to define the human dialogue partner (default "### human:")
    format_assistant : str
        a formatted string to define the assistant dialogue partner(default "### assistant:")
    column_prompt : str
        a formatted string to define in which DataFrame column the prompt can be found (default "instruction")
    column_response : str
        a formatted string to define in which DataFrame column the response can be found (default "response")
    text_separator : str
        a formatted string to define text separator (default " ")
    max_text_chunks : int
        an integer to define maximum number of text chunks per training item (default 2000)
    """

    def __init__(self,
                 input_dir: str = "./data/datasets_to_process",
                 eval_ds: List[str] = [],
                 instruction_prompt: str = default_instruction_prompt,
                 format_human: str = "### human:",
                 format_assistant: str = "### assistant:",
                 column_prompt: str = "instruction",
                 column_response: str = "response",
                 text_separator: str = " ",
                 max_text_chunks: int = 10000,
                 test_size: float = 0.15) -> None:
        
        self.input_dir = input_dir
        self.instruction_prompt = instruction_prompt
        self.format_human = format_human
        self.format_assistant = format_assistant
        self.column_prompt = column_prompt
        self.column_response = column_response
        self.text_separator = text_separator 
        self.max_text_chunks = max_text_chunks 
        self.test_size = test_size
        self.eval_ds = eval_ds
        self.prepare_datasets()

    def prepare_dataframe(self,df: pd.DataFrame) -> List[str]:

        prompt_response_list = []
        
        for _,row in df.iterrows():

            text = self.instruction_prompt + '\n'
            text += self.format_human + ' ' + row[self.column_prompt]  +'\n'
            text += self.format_assistant + ' ' + row[self.column_response]

            prompt_response_list.append(text)


        return prompt_response_list
        
    def prepare_text(self,text: str) -> List[str]:
        text_split = text.split(self.text_separator)

        prompt_list = [text_split[i:i + self.max_text_chunks] for i in range(0, len(text_split), self.max_text_chunks)] 

        prompt_list = [self.text_separator.join(chunk) for chunk in prompt_list]

        return prompt_list

    def process_pickle(self,file: str) -> List:

        with open(file, 'rb') as f:

            pickle_obj = pickle.load(f)

        if isinstance(pickle_obj,list):

            return pickle_obj
        
        elif isinstance(pickle_obj,pd.DataFrame):
            return self.prepare_dataframe(pickle_obj)
        
        else:
            print(f'{file} is not a list or a DataFrame, it will be ignored')
            return []
        
    def list_to_ds(self,ds_list: List[str], split: bool = False) -> Union[Dataset, Dict[str, Dataset]]:

        random.shuffle(ds_list)
        
        ds = Dataset.from_dict({'text': ds_list})
        if split:
            return ds.train_test_split(test_size=self.test_size)
        else:
            return ds

    def read_file(self, file: str, extension: str) -> Union[List[str], None]:

        if extension == 'pkl' or extension == 'pickle':
            return self.process_pickle(file)

        elif extension == 'txt':
            return self.prepare_text(file)

        else:
            print(f'{file} is not a list or a DataFrame, it will be ignored')
            return None
        

    def combine_datasets(self) -> Tuple[Dict[str, Dataset], Dict[str, Dataset]]:

        files = glob.glob(self.input_dir+"/*")

        if not self.eval_ds:
            self.eval_ds = [file.split('/')[-1] for file in files]
       
        ds_dict_train = {}
        ds_dict_test = {}
        for file in files: 

            extension = file.split('/')[-1].split('.')[1]
            fname = file.split('/')[-1]

            ds_list = self.read_file(file,extension)
            ds_name = fname.split('.')[0]
            if fname in self.eval_ds:
                ds = self.list_to_ds(ds_list,split=True)
                ds_dict_train[ds_name] = ds['train']
                
                ds_dict_test[ds_name] = ds['test']
            else:
                ds = self.list_to_ds(ds_list,split=False)
                ds_dict_train[ds_name] = ds

        return ds_dict_train, ds_dict_test

    def prepare_datasets(self) -> None:

        self.train_dataset, self.test_dataset = self.combine_datasets()


        #self.train_dataset = concatenate_datasets(ds_list_train).shuffle(seed=42)
        #self.test_dataset = concatenate_datasets(ds_dict_test).shuffle(seed=42)

def save_ds_list(ds_list, folder,suffix=""):
    if suffix:
        suffix ='_'+suffix
    for ds_name in ds.test_dataset.keys():
        with open(folder+'/'+ds_name+suffix+'.pkl', 'wb') as handle:
            pickle.dump(ds_list[ds_name], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    ds = PrepareDataset()
    print(ds.train_dataset)
    print(ds.test_dataset)

    train_folder = 'data/dataset_processed/train'
    eval_folder = 'data/dataset_processed/eval'
   
    save_ds_list(ds.train_dataset,train_folder,suffix='train')
    save_ds_list(ds.test_dataset,eval_folder,suffix='eval')

    text = ''
    for item in ds.test_dataset.values():

        for txt in item:
            text += txt['text'] +'\n\n'



    with open('test.txt', 'w') as file:
        file.write(text)

    

