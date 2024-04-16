import pandas as pd
import numpy as np
import torch
import re
import unicodedata
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
import gensim.downloader
from Levenshtein import distance as levenshtein_dist
from src.imputer import DataFrameImputer

def remove_accents(input_str):
    """Remove accents from a string.

    Args:
        input_str (str): The input string.

    Returns:
        str: The string without accents.
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def remove_uppercase(input_str):
    """Converts a string to lowercase.

    Args:
        input_str (str): The input string.

    Returns:
        str: The string in lowercase.
    """
    return input_str.lower()

def remove_periods(input_str):
    """Remove periods from a string.

    Args:
        input_str (str): The input string.

    Returns:
        str: The string without periods.
    """
    return input_str.replace('.', '')

def remove_de_d_du(input_str):
    """Remove "de", "d", "du" words from a string.

    Args:
        input_str (str): The input string.

    Returns:
        str: The string without "de", "d", "du" words.
    """
    return re.sub(r'\b(de|d|du)\b', "", input_str)

def remove_spaces(input_str):
    """Remove spaces from a string.

    Args:
        input_str (str): The input string.

    Returns:
        str: The string without spaces.
    """
    return input_str.replace(' ', '')

def pipeline_remove(input_str):
    """Apply a series of string cleaning steps to the input string.

    Args:
        input_str (str): The input string.

    Returns:
        str: The cleaned string.
    """
    return remove_spaces(remove_de_d_du(remove_periods(remove_uppercase(remove_accents(input_str)))))


def convert_string_age_to_float(str_age):
    """Convert string age variable to float with regex.

    Args:
        str_age (str): Age character string.

    Returns:
        float: Float age or NaN.
    """
    #jour(s), #semaine(s), #semmaine(s), #mois, #an(s), #annee(s)
    str_age = remove_accents(remove_uppercase(remove_periods(remove_spaces(str_age))))
    try: #age='52'
        age = float(str_age)
    except:
        if bool(re.search(r'jour|jours', str_age)): #contient jour(s)
            str_age_without_jour = re.sub(r'jour(s)', '', str_age)

            if bool(re.search(r'etdemi', str_age)): #contient "etdemi"
                str_age_without_jour = re.sub(r'etdemi', '', str_age_without_jour)

            elif bool(re.search(r'demi', str_age)): #contient "demi"
                str_age_without_jour = re.sub(r'demi', '', str_age_without_jour)

            try:
                age = float(str_age_without_jour)/365
            except:
                age = np.nan

        elif bool(re.search(r'semaine|semaines', str_age)): #contient semaine(s)
            str_age_without_sem = re.sub(r'semaine(s)', '', str_age)

            if bool(re.search(r'etdemi', str_age)): #contient "etdemi"
                str_age_without_sem = re.sub(r'etdemi', '', str_age_without_sem)

            elif bool(re.search(r'demi', str_age)): #contient "demi"
                str_age_without_sem = re.sub(r'demi', '', str_age_without_sem)
            try:
                age = float(str_age_without_sem)/52.143
            except:
                age = np.nan

        elif bool(re.search(r'semmaine|semmainess', str_age)): #contient semmaine(s)
            str_age_without_sem = re.sub(r'semmaine(s)', '', str_age)

            if bool(re.search(r'etdemi', str_age)): #contient "etdemi"
                str_age_without_sem = re.sub(r'etdemi', '', str_age_without_sem)

            elif bool(re.search(r'demi', str_age)): #contient "demi"
                str_age_without_sem = re.sub(r'demi', '', str_age_without_sem)
            try:
                age = float(str_age_without_sem)/52.143
            except:
                age = np.nan

        elif bool(re.search(r'mois', str_age)): #contient mois
            str_age_without_mois= re.sub(r'mois', '', str_age)

            if bool(re.search(r'etdemi', str_age)): #contient "etdemi"
                str_age_without_mois = re.sub(r'etdemi', '', str_age_without_mois)
                try:
                    age = float(str_age_without_mois)/12 + 1/24
                except:
                    age = np.nan

            elif bool(re.search(r'demi', str_age)): #contient "demi"
                str_age_without_mois = re.sub(r'demi', '', str_age_without_mois)
                try:
                    age = float(str_age_without_mois)/12 + 1/24
                except:
                    age = np.nan
            else:
                try:
                    age = float(str_age_without_mois)/12
                except:
                    age = np.nan
        
        elif bool(re.search(r'an|an(s)', str_age)): #contient an(s)
            str_age_without_ans= re.sub(r'an(s)', '', str_age)

            if bool(re.search(r'etdemi', str_age)): #contient "etdemi"
                str_age_without_ans = re.sub(r'etdemi', '', str_age_without_ans)
                try:
                    age = float(str_age_without_ans) + 0.5
                except:
                    age = np.nan

            elif bool(re.search(r'demi', str_age)): #contient "demi"
                str_age_without_ans = re.sub(r'demi', '', str_age_without_ans)
                try:
                    age = float(str_age_without_ans)
                except:
                    age = np.nan
            else:
                try:
                    age = float(str_age_without_ans)
                except:
                    age = np.nan

        elif bool(re.search(r'annee|annees', str_age)): #contient annee(s)
            str_age_without_ans= re.sub(r'annee(s)', '', str_age)

            if bool(re.search(r'etdemi', str_age)): #contient "etdemi"
                str_age_without_ans = re.sub(r'etdemi', '', str_age_without_ans)
                try:
                    age = float(str_age_without_ans) + 0.5
                except:
                    age = np.nan

            elif bool(re.search(r'demi', str_age)): #contient "demi"
                str_age_without_ans = re.sub(r'demi', '', str_age_without_ans)
                try:
                    age = float(str_age_without_ans)
                except:
                    age = np.nan
            else:
                try:
                    age = float(str_age_without_ans)
                except:
                    age = np.nan
        else:
            age = np.nan
    return age 

def convert_string_birth_date_to_int(str_birth_date):
    """Convert string birth date variable to float.

    Args:
        str_birth_date (str): Birth date character string.

    Returns:
        float: Float birth date or NaN.
    """
    try:
        birth_date = int(str_birth_date)
        if (birth_date>=1700)&(birth_date<=1936):
            return birth_date
        else:
            return np.nan
    except:
        return np.nan

def find_nationality(token_nationality, dict_countries):
    """Assigns a nationality category.

    Args:
        token_nationality (list): tokenized nationality.
        dict_countries (dict): key is country, value is nationality name.

    Returns:
        str or float: Nationality category or Nan.
    """
    for word in token_nationality:
        if word == 'unk':
            return np.nan
        else:
            for key, value in dict_countries.items():
                if isinstance(value, list):
                    for value_i in value:
                        if levenshtein_dist(word, key)<=2 or levenshtein_dist(word, value_i) <=2:
                            return key
                else:
                    if levenshtein_dist(word, key)<=2 or levenshtein_dist(word, value) <=2:
                        return key
    return "autre"     

def get_mean_vector(w2v_vectors, words):
    """Calculates the average word embedding of words in a document

    Args:
        w2v_vectors (dict): A dictionary where keys are words and values are their corresponding word embeddings.
        words (list): A list of words in the document.

    Returns:
        numpy.ndarray: The average word embedding vector of the words in the document.
    """
    words = [word for word in words if word in w2v_vectors]
    if words:
        avg_vector = np.mean(w2v_vectors[words], axis=0)
    else:
        avg_vector = np.zeros_like(w2v_vectors['hi'])
    return avg_vector


class PreprocessDataTrainTestSplit:
    """Train test split and preprocessing of text dataset
    """
    def __init__(self, df, index_test = 17814, col_to_delete = ["observation", "employer", "lob"]):
        """Class constructor

        Args:
            df (pandas.DataFrame): text dataset.
            index_test (int, optional): first index of test dataset. Defaults to 17814.
            col_to_delete (list, optional): columns to delete. Defaults to ["observation", "employer", "lob"].
        """
        self.df = df
        self.index_test = index_test
        self.col_to_delete = col_to_delete
    
    def run(self):
        """Run preprocessing and train test split

        Returns:
            Tuple[pandas.DataFrame, pandas.DataFrame]: train and test dataframe
        """
        ##1: drop columns
        self.df = self.df.drop(columns = self.col_to_delete)

        ##2: train-test split
        does_first_row_contains_idem = True
        while does_first_row_contains_idem:
            first_row_test = self.df.iloc[self.index_test].values
            num_val_idem = 0
            for val in first_row_test:
                if isinstance(val, str):
                    check = re.search(r'idem', val , flags=re.IGNORECASE)
                    if check is not None :
                        num_val_idem += 1
            if num_val_idem > 0:
                self.index_test = self.index_test -1
            else:
                does_first_row_contains_idem = False
        print("Final test index: {}".format(self.index_test))

        train_df = self.df.iloc[:self.index_test]
        test_df = self.df.iloc[self.index_test:]

        ##3: replace drop rows with full NaN
        len_train = len(train_df)
        len_test = len(test_df)

        train_df = train_df.dropna(subset=['surname', 'firstname', 'occupation', 'age', 'civil_status', 'nationality', 'link', 'birth_date'], how='all')
        test_df = test_df.dropna(subset=['surname', 'firstname', 'occupation', 'age', 'civil_status', 'nationality', 'link', 'birth_date'], how='all')

        print("Number of rows with only NaN - Train: {}".format(len_train - len(train_df)))
        print("Number of rows with only NaN - Test: {}".format(len_test - len(test_df)))

        ##4: replace "idem" by the previous value
        for i in range(1, len(train_df)):
            for j in range(len(train_df.columns)):
                cell_value = train_df.iloc[i, j]
                if isinstance(cell_value, str) and re.search(r'idem', cell_value, flags=re.IGNORECASE):
                    train_df.iloc[i, j] = train_df.iloc[i - 1, j]
        
        for i in range(1, len(test_df)):
            for j in range(len(test_df.columns)):
                cell_value = test_df.iloc[i, j]
                if isinstance(cell_value, str) and re.search(r'idem', cell_value, flags=re.IGNORECASE):
                    test_df.iloc[i, j] = test_df.iloc[i - 1, j]
        
        return train_df, test_df
    
class EmbeddingData:
    """Embedding and cleaning class
    """
    def __init__(self, df_train, df_test, col_age, col_civil_status, col_birth_date, col_nationality, col_occupation, col_firstname, col_surname) -> None:
        """Initializes an EmbeddingData object.

        Args:
            df_train (pandas.DataFrame): The training DataFrame.
            df_test (pandas.DataFrame): The test DataFrame.
            col_age (str): The column name for age.
            col_civil_status (str): The column name for civil status.
            col_birth_date (str): The column name for birth date.
            col_nationality (str): The column name for nationality.
            col_occupation (str): The column name for occupation.
            col_firstname (str): The column name for first name.
            col_surname (str): The column name for surname.
        """
        self.df_train = df_train
        self.df_test = df_test

        self.df_train_clean = None
        self.df_test_clean = None
        self.is_df_train_clean = False
        self.is_df_test_clean = False

        self.col_age = col_age
        self.col_civil_status = col_civil_status
        self.col_birth_date = col_birth_date
        self.col_nationality = col_nationality
        self.col_occupation = col_occupation
        self.col_firstname = col_firstname
        self.col_surname = col_surname

        self.unique_train_civil_values = None
        self.unique_train_nat_values = None

        self.imputer = DataFrameImputer()
        self.label_encoder_civil = LabelEncoder()
        self.label_encoder_nat = LabelEncoder()

    def clean_train_dataframe(self):
        """Cleans the training DataFrame.
        """
        df_clean = self.df_train.copy()
        columns_to_fill = df_clean.columns.difference([self.col_civil_status])
        df_clean[columns_to_fill] = df_clean[columns_to_fill].fillna('UNK')
        stop_words = set(stopwords.words('french'))

        #1: Age
        df_clean[self.col_age] = df_clean[self.col_age].apply(convert_string_age_to_float)

        #2: Birth date
        df_clean[self.col_birth_date] = df_clean[self.col_birth_date].apply(convert_string_birth_date_to_int)

        #3: Nationality
        EU_pays = {
            "allemagne": "allemand",
            "autriche": "autrichien",
            "belgique": "belge",
            "bulgarie": "bulgare",
            "chypre": "chypriote",
            "croatie": "croate",
            "danemark": "danois",
            "espagne": "espagnol",
            "estonie": "estonien",
            "finlande": "finlandais",
            "france": "francais",
            "grece": "grec",
            "hongrie": "hongrois",
            "irlande": "irlandais",
            "italie": "italien",
            "lettonie": "letton",
            "lituanie": "lituanien",
            "luxembourg": "luxembourgeois",
            "malte": "maltais",
            "pays-bas": ["neerlandais", "hollandais"],
            "pologne": "polonais",
            "portugal": "portugais",
            "republique tcheque": "tcheque",
            "roumanie": "roumain",
            "slovaquie": "slovaque",
            "slovenie": "slovene",
            "suede": "suedois",
            "royaume-uni": "anglais",
            "unk": "unk"
        }

        array_nationality = df_clean[self.col_nationality].values
        array_nationality_token  = [nltk.word_tokenize(nat, language='french') for nat in array_nationality]
        clean_nationality_token = [[remove_uppercase(remove_accents(remove_periods(word))) for word in token] for token in array_nationality_token]
        clean_nationality_token_without_stop = [[w for w in token if not w in stop_words] for token in clean_nationality_token]
        df_clean[self.col_nationality] = np.array([find_nationality(token, EU_pays) for token in clean_nationality_token_without_stop])

        #4 Occupation
        lemm = WordNetLemmatizer()
        array_occupation = df_clean[self.col_occupation].values
        token_occupation = [nltk.word_tokenize(occ, language='french') for occ in array_occupation]
        clean_token_occupation = [[remove_accents(remove_uppercase(remove_periods(remove_spaces(word)))) for word in occupation if word] for occupation in token_occupation]
        clean_token_occupation_without_stop = [[w for w in occupation if not w in stop_words] for occupation in clean_token_occupation]
        lemma_occupation = [[lemm.lemmatize(w) for w in occupation] for occupation in clean_token_occupation_without_stop]
        for i in range(len(lemma_occupation)):
            for j in range(len(lemma_occupation[i])):
                if lemma_occupation[i][j] == "unk":
                    lemma_occupation[i][j] = "UNK"
        df_clean[self.col_occupation] = lemma_occupation

        #5 First name
        to_delete_name = ['d', 'épicière', 'te', 'vve', 'n', 'femme', 'veuve', 'ep', 've']
        array_firstname = df_clean[self.col_firstname].values
        token_firstname = [nltk.word_tokenize(name, language='french') for name in array_firstname]
        token_firstname_clean = [[word for word in token if word not in to_delete_name] for token in token_firstname]
        for i in range(len(token_firstname_clean)):
            if len(token_firstname_clean[i]) > 1:
                res = [' '.join(token_firstname_clean[i])]
                token_firstname_clean[i] = res
        df_clean[self.col_firstname] = token_firstname_clean
        
        #6 Surname
        array_surname = df_clean[self.col_surname].values
        token_surname = [nltk.word_tokenize(surname, language='french') for surname in array_surname]
        token_surname_clean = [[word for word in token if re.match(r'^[A-Z]', word)] for token in token_surname]
        for i in range(len(token_surname_clean)):
            if len(token_surname_clean[i]) > 1:
                res = [' '.join(token_surname_clean[i])]
                token_surname_clean[i] = res
        df_clean[self.col_surname] = token_surname_clean

        #Imputation Age, Birth Date, Nationaly and Civil Status
        columns_to_impute = [self.col_age, self.col_nationality, self.col_civil_status, self.col_birth_date]
        df_to_impute = df_clean[columns_to_impute]

        imputed_values = self.imputer.fit_transform(df_to_impute)
        imputed_DF = pd.DataFrame(imputed_values, columns=columns_to_impute, index=df_clean.index)

        for column in df_clean.columns:
            if column not in columns_to_impute:
                imputed_DF[column] = df_clean[column]

        #Encoding Civil status 
        self.unique_train_civil_values = imputed_DF[self.col_civil_status].unique() #Unique civil status in train set
        self.label_encoder_civil.fit(imputed_DF[self.col_civil_status])
        imputed_DF[self.col_civil_status] = self.label_encoder_civil.transform(imputed_DF[self.col_civil_status])

        #Encoding Nationality
        self.unique_train_nat_values = imputed_DF[self.col_nationality].unique() #Unique nationality values in train set
        self.label_encoder_nat.fit(imputed_DF[self.col_nationality])
        imputed_DF[self.col_nationality] = self.label_encoder_nat.transform(imputed_DF[self.col_nationality])

        self.df_train_clean = imputed_DF
        self.is_df_train_clean = True

    def embedding_train_dataframe(self, name_model_embedding, model_embedding = None, tokenizer = None):
        """Creates embeddings for the training DataFrame.

        Args:
            name_model_embedding (str): The name of the embedding model: 'word2vec' or 'glove' or 'camemBERT'
            model_embedding (object, optional): The embedding model object. Defaults to None.
            tokenizer (object, optional): The tokenizer object. Defaults to None.

        Returns:
            numpy.ndarray: The embedded features for the training DataFrame.
            
        Raises:
            ValueError: If the training DataFrame has not been cleaned.
        """
        if not self.is_df_train_clean:
            raise ValueError("Le DataFrame Train n'a pas été nettoyé préalablement.")
        
        if name_model_embedding == 'word2vec':
            X_tokens = [self.df_train_clean[self.col_surname].iloc[i] + self.df_train_clean[self.col_firstname].iloc[i] + self.df_train_clean[self.col_occupation].iloc[i] for i in range(len(self.df_train_clean))]
            w2v_model = Word2Vec(X_tokens, vector_size=200, window=5, min_count=1, workers=4)
            X_vectors = np.array([get_mean_vector(w2v_model.wv, words)for words in X_tokens])
            X_others = self.df_train_clean[[self.col_age, self.col_birth_date, self.col_nationality, self.col_civil_status]]
            X = np.concatenate((X_vectors, X_others), axis=1)
        
        elif name_model_embedding == 'glove':
            X_tokens = [self.df_train_clean[self.col_surname].iloc[i] + self.df_train_clean[self.col_firstname].iloc[i] + self.df_train_clean[self.col_occupation].iloc[i] for i in range(len(self.df_train_clean))]
            X_vectors = np.array([get_mean_vector(model_embedding, words) for words in X_tokens])
            X_others = self.df_train_clean[[self.col_age, self.col_birth_date, self.col_nationality, self.col_civil_status]]
            X = np.concatenate((X_vectors, X_others), axis=1)
        
        elif name_model_embedding == 'camemBERT':
            X_tokens = [self.df_train_clean[self.col_surname].iloc[i] + self.df_train_clean[self.col_firstname].iloc[i] + self.df_train_clean[self.col_occupation].iloc[i] for i in range(len(self.df_train_clean))]
            X_tokens = [' '.join(token) for token in X_tokens]
            phrases_tokens = tokenizer(X_tokens, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model_embedding(**phrases_tokens)
                last_hidden_states = outputs.last_hidden_state
            X_vectors = last_hidden_states[:, 0, :].numpy()
            X_others = self.df_train_clean[[self.col_age, self.col_birth_date, self.col_nationality, self.col_civil_status]]
            X = np.concatenate((X_vectors, X_others), axis=1)

        return X
    
    def no_embedding_train_dataframe(self):
        """Returns features without embedding for the training DataFrame.

        Returns:
            pandas.DataFrame: The features without embedding for the training DataFrame.
        """
        X = self.df_train_clean[[self.col_age, self.col_birth_date, self.col_nationality, self.col_civil_status]]
        return X

    def clean_test_dataframe(self):
        """Cleans the test DataFrame.

        Raises:
            ValueError: If the training DataFrame has not been cleaned.
        """
        if not self.is_df_train_clean:
            raise ValueError("Le DataFrame Train n'a pas été nettoyé préalablement.")
        
        df_clean = self.df_test.copy()
        columns_to_fill = df_clean.columns.difference([self.col_civil_status])
        df_clean[columns_to_fill] = df_clean[columns_to_fill].fillna('UNK')
        stop_words = set(stopwords.words('french'))

        #1: Age
        df_clean[self.col_age] = df_clean[self.col_age].apply(convert_string_age_to_float)

        #2: Birth date
        df_clean[self.col_birth_date] = df_clean[self.col_birth_date].apply(convert_string_birth_date_to_int)

        #3: Nationality
        EU_pays = {
            "allemagne": "allemand",
            "autriche": "autrichien",
            "belgique": "belge",
            "bulgarie": "bulgare",
            "chypre": "chypriote",
            "croatie": "croate",
            "danemark": "danois",
            "espagne": "espagnol",
            "estonie": "estonien",
            "finlande": "finlandais",
            "france": "francais",
            "grece": "grec",
            "hongrie": "hongrois",
            "irlande": "irlandais",
            "italie": "italien",
            "lettonie": "letton",
            "lituanie": "lituanien",
            "luxembourg": "luxembourgeois",
            "malte": "maltais",
            "pays-bas": ["neerlandais", "hollandais"],
            "pologne": "polonais",
            "portugal": "portugais",
            "republique tcheque": "tcheque",
            "roumanie": "roumain",
            "slovaquie": "slovaque",
            "slovenie": "slovene",
            "suede": "suedois",
            "royaume-uni": "anglais",
            "unk": "unk"
        }

        array_nationality = df_clean[self.col_nationality].values
        array_nationality_token  = [nltk.word_tokenize(nat, language='french') for nat in array_nationality]
        clean_nationality_token = [[remove_uppercase(remove_accents(remove_periods(word))) for word in token] for token in array_nationality_token]
        clean_nationality_token_without_stop = [[w for w in token if not w in stop_words] for token in clean_nationality_token]
        df_clean[self.col_nationality] = np.array([find_nationality(token, EU_pays) for token in clean_nationality_token_without_stop])

        #4 Occupation
        lemm = WordNetLemmatizer()
        array_occupation = df_clean[self.col_occupation].values
        token_occupation = [nltk.word_tokenize(occ, language='french') for occ in array_occupation]
        clean_token_occupation = [[remove_accents(remove_uppercase(remove_periods(remove_spaces(word)))) for word in occupation if word] for occupation in token_occupation]
        clean_token_occupation_without_stop = [[w for w in occupation if not w in stop_words] for occupation in clean_token_occupation]
        lemma_occupation = [[lemm.lemmatize(w) for w in occupation] for occupation in clean_token_occupation_without_stop]
        for i in range(len(lemma_occupation)):
            for j in range(len(lemma_occupation[i])):
                if lemma_occupation[i][j] == "unk":
                    lemma_occupation[i][j] = "UNK"
        df_clean[self.col_occupation] = lemma_occupation

        #5 First name
        to_delete_name = ['d', 'épicière', 'te', 'vve', 'n', 'femme', 'veuve', 'ep', 've']
        array_firstname = df_clean[self.col_firstname].values
        token_firstname = [nltk.word_tokenize(name, language='french') for name in array_firstname]
        token_firstname_clean = [[word for word in token if word not in to_delete_name] for token in token_firstname]
        for i in range(len(token_firstname_clean)):
            if len(token_firstname_clean[i]) > 1:
                res = [' '.join(token_firstname_clean[i])]
                token_firstname_clean[i] = res
        df_clean[self.col_firstname] = token_firstname_clean
        
        #6 Surname
        array_surname = df_clean[self.col_surname].values
        token_surname = [nltk.word_tokenize(surname, language='french') for surname in array_surname]
        token_surname_clean = [[word for word in token if re.match(r'^[A-Z]', word)] for token in token_surname]
        for i in range(len(token_surname_clean)):
            if len(token_surname_clean[i]) > 1:
                res = [' '.join(token_surname_clean[i])]
                token_surname_clean[i] = res
        df_clean[self.col_surname] = token_surname_clean

        #Imputation Age, Birth Date, Nationaly and Civil status
        columns_to_impute = [self.col_age, self.col_nationality, self.col_civil_status, self.col_birth_date]
        df_to_impute = df_clean[columns_to_impute]
        ##NaN if there are values in test column civil status wich are not in train
        df_to_impute.loc[~df_to_impute[self.col_civil_status].isin(self.unique_train_civil_values), self.col_civil_status] = np.nan
        ##NaN if there are values in test column nationality wich are not in train
        df_to_impute.loc[~df_to_impute[self.col_nationality].isin(self.unique_train_nat_values), self.col_nationality] = np.nan

        imputed_values = self.imputer.transform(df_to_impute)
        imputed_DF = pd.DataFrame(imputed_values, columns=columns_to_impute, index=df_clean.index)

        for column in df_clean.columns:
            if column not in columns_to_impute:
                imputed_DF[column] = df_clean[column]

        #Encoding civil status
        imputed_DF[self.col_civil_status] = self.label_encoder_civil.transform(imputed_DF[self.col_civil_status])
        
        #Encoding Nationality
        imputed_DF[self.col_nationality] = self.label_encoder_nat.transform(imputed_DF[self.col_nationality])
        
        self.df_test_clean = imputed_DF
        self.is_df_test_clean = True

    def embedding_test_dataframe(self, name_model_embedding, model_embedding = None, tokenizer = None):
        """Creates embeddings for the test DataFrame.

        Args:
            name_model_embedding (str): The name of the embedding model.
            model_embedding (object, optional): The embedding model object. Defaults to None.
            tokenizer (object, optional): The tokenizer object. Defaults to None.

        Returns:
            numpy.ndarray: The embedded features for the test DataFrame.
            
        Raises:
            ValueError: If the test DataFrame has not been cleaned.
        """
        if not self.is_df_test_clean:
            raise ValueError("Le DataFrame Test n'a pas été nettoyé préalablement.")
        
        if name_model_embedding == 'word2vec':
            X_tokens = [self.df_test_clean[self.col_surname].iloc[i] + self.df_test_clean[self.col_firstname].iloc[i] + self.df_test_clean[self.col_occupation].iloc[i] for i in range(len(self.df_test_clean))]
            w2v_model = Word2Vec(X_tokens, vector_size=200, window=5, min_count=1, workers=4)
            X_vectors = np.array([get_mean_vector(w2v_model.wv, words)for words in X_tokens])
            X_others = self.df_test_clean[[self.col_age, self.col_birth_date, self.col_nationality, self.col_civil_status]]
            X = np.concatenate((X_vectors, X_others), axis = 1)

        elif name_model_embedding == 'glove':
            X_tokens = [self.df_test_clean[self.col_surname].iloc[i] + self.df_test_clean[self.col_firstname].iloc[i] + self.df_test_clean[self.col_occupation].iloc[i] for i in range(len(self.df_test_clean))]
            X_vectors = np.array([get_mean_vector(model_embedding, words) for words in X_tokens])
            X_others = self.df_test_clean[[self.col_age, self.col_birth_date, self.col_nationality, self.col_civil_status]]
            X = np.concatenate((X_vectors, X_others), axis=1)
    
        elif name_model_embedding == 'camemBERT':
            X_tokens = [self.df_test_clean[self.col_surname].iloc[i] + self.df_test_clean[self.col_firstname].iloc[i] + self.df_test_clean[self.col_occupation].iloc[i] for i in range(len(self.df_test_clean))]
            X_tokens = [' '.join(token) for token in X_tokens]
            phrases_tokens = tokenizer(X_tokens, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model_embedding(**phrases_tokens)
                last_hidden_states = outputs.last_hidden_state
            X_vectors = last_hidden_states[:, 0, :].numpy()
            X_others = self.df_test_clean[[self.col_age, self.col_birth_date, self.col_nationality, self.col_civil_status]]
            X = np.concatenate((X_vectors, X_others), axis = 1)
        return X
    
    def no_embedding_test_dataframe(self):
        """Returns features without embedding for the test DataFrame.

        Returns:
            pandas.DataFrame: The features without embedding for the test DataFrame.
        """
        X = self.df_test_clean[[self.col_age, self.col_birth_date, self.col_nationality, self.col_civil_status]]
        return X
