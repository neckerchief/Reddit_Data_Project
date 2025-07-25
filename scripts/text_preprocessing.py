import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# ----------- CLEANING FUNCTION -----------
def clean_reddit_text(text):
    if pd.isna(text):
        return ""
    
    # 1. Remove Reddit artifacts
    text = str(text)
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)

    # 2. Remove URLs and HTML entities
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'&\w+;', ' ', text)

    # 3. Lowercase
    text = text.lower()

    # 4. Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)

    # 5. Tokenize
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    # 7. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 8. Re-join tokens
    cleaned_text = ' '.join(tokens)

    return cleaned_text


# ----------- PIPELINE FUNCTION -----------
def preprocess_dataframe(df, text_col1='title', text_col2='selftext'):
    df = df.copy()
    
    # Combine text fields
    df['full_text'] = df[text_col1].fillna('') + ' ' + df[text_col2].fillna('')
    
    # Clean text
    df['clean_text'] = df['full_text'].apply(clean_reddit_text)
    
    return df
