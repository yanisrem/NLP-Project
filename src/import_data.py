import json
import re
import pandas as pd
def import_data(path = "C:\\Users\\yanis\\OneDrive\\Documents\\ENSAE 3A\\NLP\\Projet\\data\entities.json"):
    """Function to import text data. Create a DataFrame in which each column corresponds to a tag.

    Returns:
        pandas.DataFrame: Dataset.
    """
    tokens = {
    'Ⓞ': 'surname',
    'Ⓕ': 'firstname',
    'Ⓜ': 'occupation',
    'Ⓐ': 'age',
    'Ⓒ': 'civil_status',
    'Ⓚ': 'nationality',
    'Ⓟ': 'surname_household',
    'Ⓗ': 'link',
    'Ⓘ': 'lob',
    'Ⓙ': 'maiden_name',
    'Ⓛ': 'observation',
    'Ⓓ': 'education_level',
    'Ⓔ': 'employer',
    'Ⓑ': 'birth_date'
    }

    f = open(path)
    data = json.load(f)
    corpus = []
    num_page = 0
    for page in data.values():
        num_page += 1
        split_page = page.split('\n')
        for line in split_page:
            tokens_in_line = re.findall(r'[\ⓄⒻⓂⒶⒸⓀⓅⒽⒾⒿⓁⒹⒺⒷ]', line)
            line_without_tokens = re.split('Ⓞ|Ⓕ|Ⓜ|Ⓐ|Ⓒ|Ⓚ|Ⓟ|Ⓗ|Ⓘ|Ⓙ|Ⓛ|Ⓓ|Ⓔ|Ⓑ', line)
            for i, word in enumerate(line_without_tokens):
                if word == '':
                    del line_without_tokens[i]
            line_without_tokens = [word.rstrip() for word in line_without_tokens]
            dict_line = dict(zip(tokens_in_line, line_without_tokens))
            dict_line["page"] = num_page
            corpus.append(dict_line)
            
    df_corpus = pd.DataFrame(corpus)
    df_corpus.rename(columns = tokens, inplace = True)
    df_corpus['surname'] = df_corpus['surname'].fillna(df_corpus['surname_household'])
    df_corpus['surname_household'] = df_corpus['surname_household'].notna().astype(int)
    df_corpus.rename(columns = {"surname_household": "y"}, inplace = True)
    return df_corpus