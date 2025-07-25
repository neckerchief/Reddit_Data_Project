import pandas as pd
import numpy as np
import re

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
        df[time_column] = pd.to_datetime(df[time_column], unit='s')
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

        # Exclamation marks (might indicate emotional intensity)
        df[f'{column}_exclamation_marks'] = df[column].str.count(r'!')

        # Repeated characters (might indicate emphasis or distress)
        df[f'{column}_repeated_chars'] = df[column].apply(
            lambda x: len(re.findall(r'(.)\1{2,}', x)) if x else 0
        )

    return df

def content_type_features(df):
    """Creates features about the type and nature of the post content"""
    df = df.copy()
    
    # Self post vs link post
    if 'is_self' in df.columns:
        df['is_self_post'] = df['is_self'].astype(int)
    
    # Has selftext content
    if 'selftext' in df.columns:
        df['has_selftext'] = (df['selftext'].fillna('').str.len() > 0).astype(int)
    
    # NSFW content
    if 'over_18' in df.columns:
        df['is_nsfw'] = df['over_18'].astype(int)
    
    # Title only posts (no selftext)
    if 'selftext' in df.columns and 'title' in df.columns:
        df['title_only_post'] = (
            (df['selftext'].fillna('').str.len() == 0) & 
            (df['title'].fillna('').str.len() > 0)
        ).astype(int)
    
    return df

def user_activity_features(df):
    """Creates features about user posting patterns (aggregate by user)"""
    df = df.copy()
    
    if 'author' not in df.columns:
        print("Warning: 'author' column not found. Skipping user activity features.")
        return df
    
    # Posts per user
    user_post_counts = df['author'].value_counts()
    df['user_total_posts'] = df['author'].map(user_post_counts)
    
    # User activity level categories
    df['user_activity_level'] = pd.cut(df['user_total_posts'], 
                                      bins=[0, 1, 3, 10, float('inf')], 
                                      labels=['Single', 'Low', 'Medium', 'High'],
                                      include_lowest=True)
    
    return df

def create_all_features(df):
    """Apply all feature engineering functions to the dataframe"""
    print("Creating word count features...")
    df = word_count_features(df)
    
    print("Creating time features...")
    df = time_features(df)
    
    print("Creating engagement features...")
    df = engagement_features(df)
    
    print("Creating text complexity features...")
    df = text_complexity_features(df)
    
    print("Creating content type features...")
    df = content_type_features(df)
    
    print("Creating user activity features...")
    df = user_activity_features(df)
    
    print(f"Feature engineering complete. DataFrame now has {len(df.columns)} columns:")
    new_columns = [col for col in df.columns if any(suffix in col for suffix in 
                   ['_count', '_length', '_ratio', '_log', 'is_', 'has_', '_level', 
                    'year', 'month', 'day', 'hour', 'weekday', 'season', 'time_of_day'])]
    
    print(f"New features created: {len(new_columns)}")
    for col in sorted(new_columns):
        print(f"  - {col}")
    
    return df

# For backwards compatibility and individual use
def word_count(df, text_columns=['title', 'selftext', 'full_text', 'clean_text']):
    """Backwards compatible function - calls word_count_features"""
    return word_count_features(df, text_columns)

if __name__ == '__main__':
    # Test the functions with sample data
    sample_data = {
        'title': ['Help me please!', 'Feeling down today...', 'Anyone else???'],
        'selftext': ['I really need help with depression.', '', 'This is a longer text with multiple sentences. How are you feeling?'],
        'score': [10, 5, -2],
        'num_comments': [15, 0, 3],
        'created_utc': [1642680000, 1642683600, 1642687200],  # Sample timestamps
        'author': ['user1', 'user2', 'user1'],
        'is_self': [True, True, True],
        'over_18': [False, False, True]
    }
    
    test_df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(test_df.info())
    
    result_df = create_all_features(test_df)
    print(f"\nFinal DataFrame shape: {result_df.shape}")

    




