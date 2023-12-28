import os
import pandas as pd
import itertools
from collections import Counter, defaultdict

def generate_ngrams(s, n=3):
    """Generate word-based n-grams from a string."""
    words = s.lower().split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def find_common_ngrams(strings, n=3):
    """Find strings with common n-grams."""
    ngrams_lists = [generate_ngrams(string, n) for string in strings]
    common = Counter(itertools.chain(*ngrams_lists)).most_common()
    common_ngrams = {item for item, count in common if count > 1}

    return [s for s in strings if any(ng in generate_ngrams(s, n) for ng in common_ngrams)]

# Path to the folder containing dataframes
folder_path = './check_duplicates'

# List all files in the folder
dataframe_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.pkl')]

# Mapping of strings to dataframes
string_to_dfs = defaultdict(set)

# Read each dataframe and process
all_strings = []
for file in dataframe_files:
    df_path = os.path.join(folder_path, file)
    df = pd.read_csv(df_path) if file.endswith('.csv') else pd.read_pickle(df_path)
    if 'examp_excerpts' in df.columns:
        df_strings = df['examp_excerpts'].dropna().tolist()
        all_strings.extend(df_strings)

        for string in df_strings:
            string_to_dfs[string].add(file)

# Find strings with common n-grams
strings_with_common_ngrams = find_common_ngrams(all_strings, n=8)

for string in strings_with_common_ngrams:
    print(f'"{string}" found in dataframes: {list(string_to_dfs[string])}')

