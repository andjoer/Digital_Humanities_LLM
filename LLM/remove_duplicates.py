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

def remove_duplicates(folder_path, column, n=3):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
    file_paths = ['./check_duplicates/gpt4_annotated_examples_v1_cleaned.pkl','./check_duplicates/gpt4_annotated_examples_2_cleaned.pkl','./check_duplicates/gpt4_human_annotated_examples_cleaned.pkl']
    datasets = {os.path.basename(path): load_dataset(path) for path in file_paths}

    compared_pairs = []
    removed_entries = {filename: set() for filename in datasets.keys()}

    for filename, df in list(datasets.items())[:-1]:  # Skip the last dataframe
        for other_filename, other_df in datasets.items():
            if filename != other_filename and (other_filename, filename) not in compared_pairs:
                for index, row in df.iterrows():
                    if index in removed_entries[filename]:  # Skip already removed entries
                        continue

                    text = str(row[column])
                    if pd.notnull(text):
                        text_ngrams = generate_ngrams(text, n)
                        for other_index, other_row in other_df.iterrows():
                            other_text = str(other_row[column])
                            if pd.notnull(other_text):
                                other_text_ngrams = generate_ngrams(other_text, n)
                                if text_ngrams & other_text_ngrams:
                                    print(f'Removing duplicate: {text} from {filename} and {other_filename}')
                                    removed_entries[filename].add(index)
                                    break
                compared_pairs.append((filename, other_filename))

    # Remove identified duplicates
    for filename, indices in removed_entries.items():
        print(filename)
        print(len(datasets[filename]))
        datasets[filename] = datasets[filename].drop(list(indices))
        print(len(datasets[filename]))
    # Optionally save the cleaned datasets
    for filename, df in datasets.items():
        cleaned_dataset_path = os.path.join(folder_path, filename.replace('.pkl', '_cleaned.pkl'))
        with open(cleaned_dataset_path, 'wb') as file:
            pickle.dump(df, file)


folder_path = './check_duplicates'  # replace with your actual folder path
remove_duplicates(folder_path, 'examp_excerpts', n=15)
