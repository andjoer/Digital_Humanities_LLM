import pandas as pd 

def read_data_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        return "File not found. Please check the file path."
    
def parse_text_to_df(data):
    # Splitting the data into rows
    rows = data.split('---')

    # Initializing a list to store the data for each row
    parsed_data = []

    for row in rows:
        # Splitting each row into two parts: excerpt and response
        parts = row.split('***')
        if len(parts) == 2:
            excerpt, response = parts
            # Adding the parsed data to the list
            parsed_data.append({
                'examp_excerpts': excerpt.strip(),
                'response': response.strip()
            })

    # Creating a DataFrame from the parsed data
    final_df = pd.DataFrame(parsed_data)
    return final_df

data = read_data_from_file('human_annotated_test.txt')
df = parse_text_to_df(data)

print(df.head())
print(len(df))

print(df.loc[1,'response'])

df.to_pickle('human_annotated_eval_gpt3.pkl')
