import pandas as pd
import numpy as np

def word_count_features(df, text_columns=['title', 'selftext', 'full_text', 'clean_text']):
    """Creates word count and text length features for specified columns"""
    df = df.copy()

    for column in text_columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in dataframe")
            continue

        # Handle NaN values
        df[column] = df[column].fillna('')

        # String length feature
        df[f'{column}_char_count'] = df[column].astype(str).str.len()

        # Word count feature
        df[f'{column}_word_count'] = df[column].astype(str).apply(
            lambda x: len(x.split()) if x.strip() else 0    # Handles empty strings
        )

        # Average word length
        df[f'{column}_avg_word_length'] = df[column].astype(str).apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.strip() else 0
        )

    return df

def time_features(df, time_column='created_utc'):
    """Creates comprehensive time-based features"""
    df = df.copy()

    if time_column not in df.columns:
        print(f"Warning: Column '{time_column}' not found in dataframe")
        return df

    # Convert to datetime
    try:
        # Unix timestamp
        df[time_column] = pd.to_datetime(df[time_column], units='s')
    except ValueError:
        try:
            # Datetime string
            df[time_column] = pd.to_datetime(df[time_column])
        except:
            print(f"Error: Could not parse {time_column} as datetime")
            return df

    df['year'] = df[time_column].dt.year
    df['month'] = df[time_column].dt.month
    df['day'] = df[time_column].dt.day
    df['hour'] = df[time_column].dt.hour
    df['weekday'] = df[time_column].dt.day_name()
    df['weekday_num'] = df[time_column].dt.dayofweek # 0=Monday, 6=Sunday

    # Categorical time features
    df['is_weekend'] = df['weekday_num'].isin([5, 6]).astype(int)

    # Time of day categories
    df['time_of_day'] = pd.cut(df['hour'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)
    
    # Season (Northern Hemisphere)
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    return df

def engagement_features(df):
    """Creates features related to post engagement and popularity"""
    df = df.copy()

    # Engagement features
    if 'score' in df.columns:
        df['score_positive'] = (df['score'] > 0 ).astype(int)
        df['score_log'] = np.sign(df['score']) * np.log1p(np.abs(df['score']))

    if 'num_comments' in df.columns:
        df['has_comments'] = (df['num_comments'] > 0).astype(int)
        df['comments_log'] = np.log1p(df['num_comments'])

    # Engagement ratio (if both score and comments exist)
    if 'score' in df.columns and 'num_comments' in df.columns:
        df['engagement_ratio'] = df['num_comments'] / np.maximum(np.abs(df['score']), 1)
    
    return df

def text_complexity_features(df, text_columns=['full_text']):
    """Creates features measuring text complexity and readability"""
    df = df.copy()

    for column in text_columns:
        if column not in df.columns:
            continue

    df[column] = df[column].fillna('')

    # Uppercase/lowercase ratios
    df[f'{column}_uppercase_ratio'] = df[column].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if x else 0
    )

    # Question marks (might indicate help-seeking behavior)
    df[f'{column}_question_marks'] = df[column].str.count(r'\?')
    

    




