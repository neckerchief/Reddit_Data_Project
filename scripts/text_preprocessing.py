import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure resources are downloaded once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Precompile outside the function
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_ENTITY_PATTERN = re.compile(r"&\w+;")
REDDIT_ARTIFACTS = re.compile(r"\[removed\]|\[deleted\]", flags=re.IGNORECASE)
DIGITS_PATTERN = re.compile(r"\d+")


# ----------- CLEANING FUNCTION -----------
def clean_reddit_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Remove Reddit artifacts, URLs, HTML entities
    text = REDDIT_ARTIFACTS.sub("", text)
    text = URL_PATTERN.sub("", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)

    # Lowercase, remove punctuation and digits
    text = text.lower().translate(PUNCTUATION_TABLE)
    text = DIGITS_PATTERN.sub("", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens if tok not in STOPWORDS and len(tok) > 2]

    return " ".join(tokens)


# ----------- PIPELINE FUNCTION -----------
def preprocess_dataframe(df, text_col1='title', text_col2='selftext'):
    df = df.copy()

    # Combine and clean text
    df['full_text'] = (
        df[text_col1].fillna('') + ' ' + df[text_col2].fillna('')
    ).astype(str)

    # Vectorized preprocessing (still row-wise, but leaner)
    df['clean_text'] = df['full_text'].apply(clean_reddit_text)

    return df
