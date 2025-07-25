import praw
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
import glob

# ----------- API CREDENTIALS -----------
# Load .env file
load_dotenv()

# Access variables
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

REDDIT = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# ----------- CONFIGURATION -----------
SUBREDDITS = ['depression', 'mentalhealth']
POST_LIMIT = 1000
SORT_BY = 'hot'  # Options: 'hot', 'new', 'top'
MASTER_FILENAME = 'reddit_posts_master.csv'  # Single file for all posts

# ----------- HELPER FUNCTION -----------
def load_existing_posts(data_dir):
    """Load existing posts from master file to avoid duplicates"""
    master_path = os.path.join(data_dir, MASTER_FILENAME)

    if os.path.exists(master_path):
        print(f"Loading existing posts from {master_path}")
        existing_df = pd.read_csv(master_path)
        existing_ids = set(existing_df['id'].tolist()) 
        print(f"Found {len(existing_ids)} existing posts IDs")
        return existing_df, existing_ids
    else:
        print("No existing master file found. Starting fresh.")
        return pd.DataFrame(), set()
    
# ----------- SCRAPING FUNCTION -----------
def scrape_subreddit(subreddit_name, sort_by='hot', limit=1000, existing_ids=None):
    """Scrape subreddit posts, filtering out existing ones"""
    if existing_ids is None:
        existing_ids = set()

    subreddit = REDDIT.subreddit(subreddit_name)
    
    if sort_by == 'new':
        posts = subreddit.new(limit=limit)
    elif sort_by == 'top':
        posts = subreddit.top(limit=limit)
    else:
        posts = subreddit.hot(limit=limit)

    records = []
    new_posts_count = 0
    skipped_posts_count = 0

    for post in posts:
        # Skip if post already exists
        if post.id in existing_ids:
            skipped_posts_count += 1
            continue

        records.append({
            'id': post.id,
            'title': post.title,
            'selftext': post.selftext,
            'score': post.score,
            'num_comments': post.num_comments,
            'created_utc': post.created_utc,
            'subreddit': subreddit_name,
            'author': str(post.author),
            'over_18': post.over_18,
            'is_self': post.is_self,
            'url': post.url,
            'scraped_at': datetime.now().isoformat()  # Track when scraped
        })
        new_posts_count += 1
    
    print(f"r/{subreddit_name}: {new_posts_count} new posts, {skipped_posts_count} duplicates skipped")
    return pd.DataFrame(records)

# ----------- SAVING FUNCTION -----------
def save_posts(all_posts, existing_posts, data_dir):
    """Save posts to master file and create timestamped backup"""
    # Combine axisitng and new posts
    if not existing_posts.empty:
        # Add scraped_at column to existing posts if it doesn't exist
        if 'scraped_at' not in existing_posts.columns:
            existing_posts['scraped_at'] = 'unknown'
        combined_posts = pd.concat([existing_posts, all_posts], ignore_index=True)
    else:
        combined_posts = all_posts

    # Remove any duplicates that might have slipped through
    combined_posts = combined_posts.drop_duplicates(subset=['id'], keep='first')

    # Save to master file
    master_path = os.path.join(data_dir, MASTER_FILENAME)
    combined_posts.to_csv(master_path, index=False)

    # Create timestamped backup of new posts only (if any new posts)
    if not all_posts.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'reddit_posts_{SORT_BY}_{timestamp}.csv'
        backup_path = os.path.join(data_dir, backup_filename)
        all_posts.to_csv(backup_path, index=False)
        print(f"New posts backup saved to {backup_path}")
    
    return combined_posts, master_path


# ----------- MAIN EXECUTION -----------
if __name__ == '__main__':
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    
    # Load existing posts
    existing_posts, existing_ids = load_existing_posts(DATA_RAW_DIR)
    
    # Scrape new posts
    new_posts = pd.DataFrame()
    total_new_posts = 0
    
    for sub in SUBREDDITS:
        print(f"Scraping r/{sub}...")
        df = scrape_subreddit(sub, SORT_BY, POST_LIMIT, existing_ids)
        if not df.empty:
            new_posts = pd.concat([new_posts, df], ignore_index=True)
            total_new_posts += len(df)
            # Update existing_ids with new posts to avoid duplicates within this run
            existing_ids.update(df['id'].tolist())
    
    # Save results
    if total_new_posts > 0:
        combined_posts, master_path = save_posts(new_posts, existing_posts, DATA_RAW_DIR)
        print(f"\n=== SCRAPING COMPLETE ===")
        print(f"New posts scraped: {total_new_posts}")
        print(f"Total posts in database: {len(combined_posts)}")
        print(f"Master file: {master_path}")
    else:
        print(f"\n=== NO NEW POSTS FOUND ===")
        print(f"All posts were duplicates. Total posts in database: {len(existing_posts)}")

