import pandas as pd

def save_to_text(pickle_path, text_filename):
    # Load the DataFrame from the pickled file
    df = pd.read_pickle(pickle_path)

    # Write the DataFrame to a text file
    with open(text_filename, 'w') as file:
        for index, row in df.iterrows():
            file.write(str(row['examp_excerpts']) + '\n***\n')
            file.write(str(row['response']) + '\n')
            file.write('---\n')  # Delimiter for each row


save_to_text('data/train_test_datasets/eval_human_gpt4_df_3/gpt4_human_annotated_examples_v3.pkl', 'human_annotated.txt')
