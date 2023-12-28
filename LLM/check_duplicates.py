import pickle
import pandas as pd
from nltk.util import ngrams
import os

def load_dataset(filename):
    with open(filename, 'rb') as file:
        dataset = pickle.load(file)
    return dataset

def generate_ngrams(text, n):
    return set([' '.join(gram) for gram in ngrams(text.split(), n)])

def find_common_entries(folder_path, column, n=3):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
    datasets = {os.path.basename(path): load_dataset(path) for path in file_paths}

    # List to keep track of already compared DataFrame pairs
    compared_pairs = []

    for filename, df in datasets.items():
        for other_filename, other_df in datasets.items():
            if filename != other_filename and (other_filename, filename) not in compared_pairs:
                for index, row in df.iterrows():
                    text = str(row[column])
                    if pd.notnull(text):
                        text_ngrams = generate_ngrams(text, n)
                        for other_index, other_row in other_df.iterrows():
                            other_text = str(other_row[column])
                            if pd.notnull(other_text):
                                other_text_ngrams = generate_ngrams(other_text, n)
                                if text_ngrams & other_text_ngrams:
                                    print('#####################')
                                    print(f'{filename} - Row {index}: {text}')
                                    print(f'{other_filename} - Row {other_index}: {other_text}')
                                    print('#####################')
                #compared_pairs.append((filename, other_filename))

# Example usage:
folder_path = 'check_duplicates'  # replace with your actual folder path
find_common_entries(folder_path, 'examp_excerpts', n=6)