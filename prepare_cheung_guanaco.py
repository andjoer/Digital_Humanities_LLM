from datasets import load_dataset
import pandas as pd
import re

from typing import List, Dict, Any

def check_lang(dct: Dict[str, str], thresh: float = 0.5) -> bool:
    """
    Check if the text in the dictionary is in a certain language based on the proportion of special characters.

    Parameters:
    dct (Dict[str, str]): Dictionary with text to check.
    thresh (float): Threshold for the ratio of special characters in text.

    Returns:
    bool: True if text is in the target language, False otherwise.
    """
    for key in dct.keys():
        if len(dct[key]) > 1 and (len(re.sub(r'[^a-zA-ZöäüÄÖÜß1-9,.!?"]','',dct[key])) / len(dct[key])) < thresh:
            return False
    return True

def chose_samples(dataset: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Choose samples from the dataset which are in the target language.

    Parameters:
    dataset (Dict[str, Any]): The dataset to choose samples from.

    Returns:
    List[Dict[str, str]]: List of samples in the target language.
    """
    chosen_samples = []
    for sample in dataset['train']:
        if check_lang(sample):
            chosen_samples.append(sample)
    return chosen_samples

def prepare_dataset(ds_list: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Prepare the dataset for training. Converts list of dictionaries into pandas DataFrame.

    Parameters:
    ds_list (List[Dict[str, str]]): List of samples in the target language.

    Returns:
    pd.DataFrame: DataFrame with instructions and responses.
    """
    instructions = []
    responses = []
    for sample in ds_list:
        if sample['input']:
            instruction = '\"'+sample['input']+'\"'+'\n\n'+sample['instruction']
        else:
            instruction = sample['instruction']
        instructions.append(instruction)
        responses.append(sample['output'])

    df = pd.DataFrame({
    "instruction": instructions,
    "response": responses
    })

    return df


if __name__ == "__main__":
    num_samples = 10000
    # The dataset is gated/private, make sure you have run huggingface-cli login
    dataset = load_dataset("JosephusCheung/GuanacoDataset")

    chosen_samples = chose_samples(dataset)
    dataset_df = prepare_dataset(chosen_samples)

    frac = num_samples/len(dataset_df)
    dataset_df = dataset_df.sample(frac=1)

    dataset_df.to_pickle('data/datasets_to_process/CheungGuanaco.pkl')