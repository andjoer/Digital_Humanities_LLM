import pickle
import argparse
from nltk.util import ngrams

def load_dataset(filename):
    with open(filename, 'rb') as file:
        dataset = pickle.load(file)
    return dataset

def generate_ngrams(text, n):
    return set([' '.join(gram) for gram in ngrams(text.split(), n)])

def split_text(text, separator,idx=-1):
    text = text.split('### assistant')[0]
    text_lst = text.split(separator)

    text = text_lst[idx]
    return text

def deduplicate_data(test_dataset_path, train_dataset_path, columns, separator, idx = -1, n=3):
    test_ds = load_dataset(test_dataset_path)
    train_ds = load_dataset(train_dataset_path)

    # Add indices as a new column
    test_ds = test_ds.add_column("index", range(len(test_ds)))

    indices_to_keep = set(range(len(test_ds)))
    for col in columns:
        if col in test_ds.features and col in train_ds.features:
            for index, row in enumerate(test_ds):
                
                test_text = split_text(str(row[col]), separator,idx=idx)

                test_ngrams = generate_ngrams(test_text, n)
                
                for train_row in train_ds:
                    train_ngrams = generate_ngrams(str(train_row[col]), n)
                    
                    if test_ngrams & train_ngrams:
                        print(f"Match found: {test_text}  #### {train_row[col]}")
                        indices_to_keep.discard(index)
                        break

    # Filter out the duplicates using the index column
    print('length of input test set: ' + str(len(test_ds)))
    test_ds = test_ds.filter(lambda example: example['index'] in indices_to_keep)
    
    # Optionally remove the index column after filtering
    test_ds = test_ds.remove_columns(["index"])
    print('length of input test set after cleaning: ' + str(len(test_ds)))
    # Save the cleaned test Dataset
    cleaned_dataset_path = test_dataset_path.replace('.pkl', '_clean.pkl')
    with open(cleaned_dataset_path, 'wb') as file:
        pickle.dump(test_ds, file)



def main():
    parser = argparse.ArgumentParser(description='Deduplicate data using n-gram matching.')
    parser.add_argument('test_dataset', type=str, help='Path to the pickled test dataset.')
    parser.add_argument('train_dataset', type=str, help='Path to the pickled train dataset.')
    parser.add_argument('--columns', nargs='+', required=True, help='List of column names to compare.')
    parser.add_argument('--max_n', type=int, default=6, help='Maximum length of n-grams (default: 3).')

    args = parser.parse_args()

    deduplicate_data(args.test_dataset, args.train_dataset, args.columns, separator='Textausschnitt:  "' ,idx = -1, n = args.max_n)

if __name__ == "__main__":
    main()


#data/train_test_datasets/run_6_onlybsp_simple/train/bsp_ds_simple_train.pkl
##data/train_test_datasets/run_6_onlybsp_simple/eval/bsp_ds_simple_eval.pkl
    
#python deduplicate.py ./data/train_test_datasets/run_4_onlybsp/eval/bsp_ds.pkl ./data/train_test_datasets/run_4_onlybsp/train/bsp_ds.pkl --columns text