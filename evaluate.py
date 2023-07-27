import pandas as pd
import pickle
import glob
from tqdm import tqdm
from functools import partial
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import transformers
import torch
import ast 
from typing import List, Dict, Any, Callable, Union, Optional, Set
import re

class Evaluate():
    """
    This class provides methods for evaluating large language models on different datasets.

    Attributes:
        model (str): Name of the model to be evaluated.
        samples_per_ds (int): Number of samples to be used for evaluation per dataset.
        tokenizer (AutoTokenizer): Tokenizer for the specified model.
        pipeline (Pipeline): Transformers pipeline for text generation.
        eval_dict (dict): Dictionary to store the evaluation results.
        statistics (dict): Dictionary to store the evaluation statistics.

    Methods:
        __init__(model, samples_per_ds): Initializes the class with the given model and sample size.
        load_ds_from_folder(folder): Loads datasets from a given folder.
        predict_responses(ds, name): Predicts responses for the given dataset and stores them in eval_dict.
        process_eval_files(folder): Loads datasets from a folder and predicts responses.
        numerical_evaluation(function_dict): Computes numerical evaluation scores for the predicted responses.
        create_score_statistics(): Computes the mean of evaluation scores for each dataset and stores them in statistics.
    """
    

    def __init__(self,
                 model: str = "",
                 samples_per_ds: int = 8):
        
        self.samples_per_ds: int = samples_per_ds
        self.model: str = model

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline: pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map={"":0},
        )

        self.eval_dict: Dict[str, List[Union[str, float]]] = {'name':[],'instruction':[],'response':[],'prediction':[],'score':[]}


    def load_ds_from_folder(self, folder: str) -> Dict[str, Any]:
        files = glob.glob(folder+"/*")
        ds = {}
        for file in files: 
            name = file.split('/')[-1].split('.')[0]
            with open(file, 'rb') as handle:
                ds[name] = pickle.load(handle)

        return ds

    def predict_responses(self, ds: Dict[str, Any], name: str) -> None:

        ds = ds['text']

        for sample in tqdm(ds[:self.samples_per_ds]):

            inst_resp = sample.split('### assistant:')

            prompt = inst_resp[0] + '### assistant:'
          
            prediction = self.pipeline(
            prompt,
            do_sample=False,
            temperature = 0.0,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens = 900,
            return_full_text=False
            )[0]['generated_text']

            self.eval_dict['instruction'].append(prompt.split('### human:')[1])
            self.eval_dict['response'].append(inst_resp[1])
            self.eval_dict['prediction'].append(prediction)
            self.eval_dict['name'].append(name)
            self.eval_dict['score'].append('-')

    def process_eval_files(self, folder: str) -> None:

        ds_dict = self.load_ds_from_folder(folder)

        for ds_name in ds_dict.keys():

            print(f'Processing Dataset {ds_name}')
            self.predict_responses(ds_dict[ds_name], ds_name)


    def numerical_evaluation(self, function_dict: Dict[str, Callable[[str, str], float]]) -> None:

        for idx, _ in enumerate(self.eval_dict['instruction']):
            name = self.eval_dict['name'][idx] 
            if name in function_dict:
                self.eval_dict['score'][idx] = function_dict[name](self.eval_dict['response'][idx], self.eval_dict['prediction'][idx])

    def create_score_statistics(self) -> None:

        score_df = pd.DataFrame.from_dict(self.eval_dict)

        names = list(set(list(score_df['name'])))

        self.statistics: Dict[str, Dict[str, float]] = {}

        for name in names: 

            name_df = score_df[score_df['name'] == name]
            
            score_dict = {}
            for _ , row in name_df.iterrows():
                if isinstance(row['score'],dict):
                    for key in row['score'].keys():
                        
                            if key not in score_dict.keys():
                                score_dict[key] = [row['score'][key]]
                            else:
                                score_dict[key].append(row['score'][key])

            for key in score_dict.keys():
                score_dict[key] = sum(score_dict[key])/len(score_dict[key])

            self.statistics[name] = score_dict

                    
def calc_common_labels_score(prediction: str, response: str, max_words: Optional[int] = None) -> float:
    """
    Calculates common labels score between prediction and response.

    Args:
        prediction (str): The predicted output.
        response (str): The actual output.
        max_words (Optional[int]): Maximum number of words to be considered. Defaults to None.

    Returns:
        float: The common labels score.
    """
    prediction = re.sub('^A-Za-zäöüßÄÖÜ',' ',prediction.lower())
    response = re.sub('^A-Za-zäöüßÄÖÜ',' ',response.lower())

    resp_labels: Set[str] = set(response.split())
    pred_labels: Set[str] = set(prediction.split())

    intersect: Set[str] = set(resp_labels) & set(pred_labels)

    divisor: int = max(len(resp_labels), len(pred_labels))

    if max_words is not None: 
        divisor = min(max_words, divisor)

    return len(intersect) / divisor

def extract_dict(string: str) -> Union[str, None]:
    """
    Extracts dictionary from the input string.

    Args:
        string (str): The input string.

    Returns:
        Union[str, None]: Extracted dictionary as a string, or None if no dictionary found.
    """
    start: int = string.find('{')
    end: int = string.find('}')
    if start != -1 and end != -1:
        return string[start:end+1]
    else: 
        return None

def get_semantic_distance(string_1: str, string_2: str, sentence_model: SentenceTransformer) -> float:
    """
    Calculates semantic distance between two strings using the provided sentence model.

    Args:
        string_1 (str): The first string.
        string_2 (str): The second string.
        sentence_model (SentenceTransformer): The sentence transformer model to be used for encoding.

    Returns:
        float: The semantic distance between the two strings.
    """
    vector_1: np.array = sentence_model.encode(string_1)
    vector_2: np.array = sentence_model.encode(string_2)

    return np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

def evaluate_examples_dict(response_dict: str, prediction_dict: str, sentence_model: SentenceTransformer) -> Union[Dict[str, float], int]:
    """
    Evaluates example-annotations by comparing prediction dictionary with response dictionary.

    Args:
        response_dict (str): The response dictionary.
        prediction_dict (str): The prediction dictionary.
        sentence_model (SentenceTransformer): The sentence transformer model to be used for encoding.

    Returns:
        Union[Dict[str, float], int]: Dictionary of scores for each key in the response dictionary, or 0 if no valid python dict.
    """
    try: 
        response_dict = ast.literal_eval(response_dict)
        prediction_dict = ast.literal_eval(prediction_dict)
    except: 
        return 0

    scores: Dict[str, float] = {}
    for key in response_dict.keys():

        prediction: str = prediction_dict.get(key)

        if prediction:

            if 'beispiel' in key.lower():
                scores[key] = get_semantic_distance(prediction, response_dict[key], sentence_model)
            else: 
                scores[key] = calc_common_labels_score(prediction, response_dict[key])

            if key.lower() == 'wiedergabeform':

                if (prediction.lower() == 'erzählstimme' or response_dict[key] == 'erzählstimme') or (prediction.lower() != 'erzählstimme' or response_dict[key] != 'erzählstimme'):
                    scores['wiedergabe_binary'] = 1
                else: 
                    scores['wiedergabe_binary'] = 0

        else: 
            scores[key] = 0
    return scores


def evaluate_examples_elastic(response: str, prediction: str, sentence_model: SentenceTransformer) -> Dict[str, float]:

    """
    Calculates total semantic distance between response and prediction.

    Args:
        response (str): The actual output.
        prediction (str): The predicted output.
        sentence_model (SentenceTransformer): The sentence transformer model to be used for encoding.

    Returns:
        dict: Dictionary containing total semantic distance.
    """


    return {'total_distance':get_semantic_distance(response,prediction,sentence_model)}


def evaluate_examples(response: str, prediction: str, sentence_model: SentenceTransformer) -> Dict:

    """
    Evaluates examples either by comparing dictionaries or calculating semantic distance.

    Args:
        response (str): The actual output.
        prediction (str): The predicted output.
        sentence_model (SentenceTransformer): The sentence transformer model to be used for encoding.

    Returns:
        dict: Dictionary of scores.
    """

    response_dict = extract_dict(response)

    if response_dict:
        score = evaluate_examples_dict(response_dict, extract_dict(prediction),sentence_model)

    else: 
        score = evaluate_examples_elastic(response,prediction,sentence_model)
    
    return score

def extract_number(string: str) -> Optional[str]:

    """
    Extracts first number from a string.

    Args:
        string (str): The string to extract the number from.

    Returns:
        str: The extracted number if found, else None.
    """

    match = re.search(r'[-+]?\d*\.\d+|\d+', string)
    if match:
        return match.group(0)
    else:
        return None

def preprocess_labeled_list(text: str) -> Dict[str, str]:

    """
    Converts a labeled list in text form to a dictionary.

    Args:
        text (str): The labeled list in text form.

    Returns:
        dict: Dictionary with number as keys and labels as values.
    """

    lines = text.split('\n')

    label_dict = {}
    for line in lines: 
        line = line.lower()
        number = extract_number(line)

        if number is not None:
            label = line.split('label:')[-1]
            label_dict[number] = label

    return label_dict

def evaluate_arguments(response: str, prediction: str) -> Dict[str, int]:

    """
    Evaluates argument-annotations by comparing response and prediction label dictionaries.

    Args:
        response (str): The actual output.
        prediction (str): The predicted output.

    Returns:
        dict: Dictionary of scores.
    """

    response_label_dict = preprocess_labeled_list(response)
    prediction_label_dict = preprocess_labeled_list(prediction)

    score_sum = 0
    score_binary_sum = 0
    for number in response_label_dict.keys():

        if prediction_label_dict.get(number):

            if prediction_label_dict[number].lower().strip() == 'o':
                if response_label_dict[number].lower().strip() == 'o':
                    score_binary = 1
                    score = 1
                else:
                    score_binary = 0
                    score = 0

            else: 
                if response_label_dict[number].lower().strip() != 'o':
                    score_binary = 1


                resp_label = prediction_label_dict[number]
                pred_label = response_label_dict[number]

                if resp_label == pred_label:
                    score = 1
                else: 
                    score = 0

        else: 
            score = 0
            score_binary = 0

        score_binary_sum += score_binary
        score_sum += score

    divisor = len(response_label_dict.keys())

    if divisor != 0:
        return {'score':score_sum/divisor, 'score binary':score_binary_sum/divisor}
    else: 
        print('divisor 0')
        print('response')
        print(response)
        print(prediction)
        return {'score':0, 'score binary':0}
    
def evaluate_redewiedergabe(response: str, prediction: str) -> Dict[str, float]:

    """
    Evaluates redewiedergabe-annotation by comparing response and prediction label dictionaries.

    Args:
        response (str): The actual output.
        prediction (str): The predicted output.

    Returns:
        dict: Dictionary of scores.
    """

    response_label_dict = preprocess_labeled_list(response)
    prediction_label_dict = preprocess_labeled_list(prediction)

    score_sum = 0
    score_binary_sum = 0
    for number in response_label_dict.keys():

        if prediction_label_dict.get(number):

            if prediction_label_dict[number].lower().strip() == 'keine redewiedergabe':
                if response_label_dict[number].lower().strip() == 'keine redewiedergabe':
                    score_binary = 1
                    score = 1
                else:
                    score_binary = 0
                    score = 0

            else: 
                if response_label_dict[number].lower().strip() != 'keine redewiedergabe':
                    score_binary = 1

                    score = calc_common_labels_score(prediction_label_dict[number],response_label_dict[number])

                else: score = 0
        else: 
            score = 0
            score_binary = 0

        score_binary_sum += score_binary
        score_sum += score

    divisor = len(response_label_dict.keys())

    return {'score':score_sum/divisor, 'score binary':score_binary_sum/divisor}


def write_readable_evaluation(eval_dict: Dict, fname: str) -> None:

    """
    Writes evaluation results in a human-readable form to a file.

    Args:
        eval_dict (dict): The evaluation dictionary.
        fname (str): The filename to write the results to.
    """

    text = ''
    for idx, _ in enumerate(eval_dict['instruction']):
        text += '# Task: ' + eval_dict['name'][idx] +' Score: ' + str(eval_dict['score'][idx])+ '\n\n'
        text += 'instruction:' + eval_dict['instruction'][idx] +'\n\n'
        text += 'response:' + eval_dict['response'][idx] +'\n\n'
        text += 'prediction:' + eval_dict['prediction'][idx] +'\n\n'
    
    with open('evaluation/'+fname+'.txt', 'w') as file:
        file.write(text)

def write_readable_statistics(statistics,fname):

    """
    Writes evaluation statistics in a human readable csv file

    Args:
        statistics (dict): The evaluation statistics.
        fname (str): The filename to write the statistics to.
    """

    datasets = list(statistics[list(statistics.keys())[0]].keys())

    experiment_score_dict = {}
    
    for name in statistics.keys():
        experiment_score_dict[name] = []
        experiment_score_dict['idx'] = []   # yes, gets overwritten every iteration
        for dataset in datasets:
            for score_name in statistics[name][dataset].keys():
                
                row_name = dataset+';'+score_name
                experiment_score_dict['idx'] .append(row_name) 
                experiment_score_dict[name].append(statistics[name][dataset][score_name])

    result_df = pd.DataFrame.from_dict(experiment_score_dict)
    result_df.index = result_df.idx   
    result_df = result_df.drop('idx', axis=1)
        
    result_df.to_csv('evaluation/'+fname+'.csv')

if __name__ == "__main__":

    sentence_model = SentenceTransformer('T-Systems-onsite/cross-en-de-roberta-sentence-transformer').to('cuda')
    evaluate_examples_sent = partial(evaluate_examples,sentence_model=sentence_model)


    eval_folder = 'data/train_test_datasets/run_1_Cheung/eval'

    models = ['results/run_1/final_merged_checkpoint/']

    statistics = {}
    for model in models: 
        
        model_name = model.split('/')[-2]
   
        eval = Evaluate(model)

        eval.process_eval_files(eval_folder)

        eval.numerical_evaluation({'redewiedergabe':evaluate_redewiedergabe,'arguments':evaluate_arguments,'bsp_ds':evaluate_examples_sent})

        write_readable_evaluation(eval.eval_dict,'evaluation_'+model_name)

        eval.create_score_statistics()

        statistics[model_name] = eval.statistics

        

    write_readable_statistics(statistics,'evaluation_statistics')