import pandas as pd
import json
import zstandard as zstd
from datetime import datetime, timezone
import calendar
import os
from pathlib import Path
import requests
from io import BytesIO
import io
import csv

DATA_DIR = Path("data/raw/zstd")
START_YEAR = 2019
START_MONTH = 1
END_YEAR = 2024
END_MONTH = 12
SUBREDDITS = ['depression', 'mentalhealth']
MASTER_FILENAME_HIST = 'reddit_posts_historical_master.csv'  # Single file for all historical posts

# ----------- USER ANONYMIZATION -----------
class UserAnonymizer:
    def __init__(self, mapping_file_path):
        self.mapping_file = mapping_file_path
        self.username_to_id = {}
        self.next_user_id = 1
        self.load_existing_mapping()

    def load_existing_mapping(self):
        """Load existing username mapping if it exists"""
        if os.path.exists(self.mapping_file):
            try:
                with open(self.mapping_file, 'r') as f:
                    data = json.load(f)
                    self.username_to_id = data.get('username_to_id', {})
                    self.next_user_id = data.get('next_user_id', 1)
                print(f"Loaded existing user mapping with {len(self.username_to_id)} users")
            except Exception as e:
                print(f"Error loading user mapping :{e}")
                print("Starting with fresh mapping")

    def save_mapping(self):
        """Save usermapping to json file"""
        os.makedirs(os.path.dirname(self.mapping_file), exist_ok=True)

        mapping_data = {
            'username_to_id': self.username_to_id,
            'next_user_id': self.next_user_id,
            'created_at': datetime.now().isoformat(),
            'total_users': len(self.username_to_id)
        }

        with open(self.mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)

    def anonymize_username(self, username):
        """COnvert username to anonymous ID"""
        # Handle delted/None users
        if username is None or username == 'None' or username == '[delted]':
            return 'user_deleted'

        username_str = str(username)

        # Check if the user occured before
        if username_str in self.username_to_id:
            return self.username_to_id[username_str]
        
        # Create new anonymous ID
        anonymous_id = f"user_{self.next_user_id:04d}"
        self.username_to_id[username_str] = anonymous_id
        self.next_user_id += 1

        return anonymous_id
    
# ----------- PARSER -----------    
def parse_zst_file(filepath, start_timestamp, end_timestamp, target_subreddit, anonymizer=None, csv_writer=None, header_written=False):
    """Parse a zstandard compressed NDJSON file and write matching records to CSV incrementally"""
    matched = 0

    try:
        with open(filepath, 'rb') as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                for line_num, line in enumerate(text_stream):
                    if not line.strip():
                        continue

                    try:
                        post = json.loads(line)

                        if post.get('subreddit', '').lower() != target_subreddit.lower():
                            continue

                        try:
                            created_utc = float(post.get('created_utc', 0))
                        except (ValueError, TypeError):
                            continue

                        if created_utc < start_timestamp or created_utc > end_timestamp:
                            continue

                        anonymous_author = anonymizer.anonymize_username(post.get('author', '[deleted]')) if anonymizer else str(post.get('author', '[deleted]'))

                        record = {
                            'id': post.get('id', ''),
                            'title': post.get('title', ''),
                            'selftext': post.get('selftext', ''),
                            'score': post.get('score', 0),
                            'num_comments': post.get('num_comments', 0),
                            'created_utc': created_utc,
                            'subreddit': target_subreddit,
                            'author': anonymous_author,
                            'over_18': post.get('over_18', False),
                            'is_self': post.get('is_self', False),
                            'scraped_at': datetime.now().isoformat()
                        }

                        # Write header if needed
                        if not header_written:
                            csv_writer.writerow(record.keys())
                            header_written = True

                        csv_writer.writerow(record.values())
                        matched += 1

                        if matched % 10000 == 0:
                            print(f"  Written {matched} matching posts...")

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
                        continue

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return matched, header_written

# ----------- EXTRACT POSTS -----------
def extract_subreddit_data_zst(subreddit_name, start_year=None, start_month=None, end_year=None, end_month=None, data_dir=None, anonymizer=None):
    """Extract and stream filtered data to CSV"""
    
    start_date = datetime(start_year, start_month, 1, tzinfo=timezone.utc)
    start_timestamp = start_date.timestamp()
    last_day = calendar.monthrange(end_year, end_month)[1]
    end_date = datetime(end_year, end_month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    end_timestamp = end_date.timestamp()
    
    print(f"Looking for {subreddit_name} posts from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    filename = f"{subreddit_name}_submissions.zst"
    filepath = os.path.join(data_dir, filename)
    out_path = os.path.join(data_dir, MASTER_FILENAME_HIST)

    header_written = os.path.exists(out_path) and os.path.getsize(out_path) > 0

    if os.path.exists(filepath):
        print(f"Processing {filepath}...")
        with open(out_path, 'a', newline='', encoding='utf-8') as out_csv:
            writer = csv.writer(out_csv)
            matched, header_written = parse_zst_file(
                filepath, start_timestamp, end_timestamp,
                target_subreddit=subreddit_name,
                anonymizer=anonymizer,
                csv_writer=writer,
                header_written=header_written
            )
    else:
        print(f"No data files found for {subreddit_name} in {data_dir}")
        return

    print(f"Done. Wrote {matched} posts to {out_path}")

# ----------- MAIN EXECUTION -----------
if __name__ == '__main__':
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    MAPPINGS_DIR = os.path.join(BASE_DIR, 'mappings')
    
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(MAPPINGS_DIR, exist_ok=True)

    # Initialize user anonymizer
    mapping_file_path = os.path.join(MAPPINGS_DIR, 'user_mapping.json')
    anonymizer = UserAnonymizer(mapping_file_path)
    
    # Scrape new posts
    new_posts = pd.DataFrame()
    total_new_posts = 0
    existing_ids = set()

    for sub in SUBREDDITS:
        extract_subreddit_data_zst(
            subreddit_name=sub,
            start_year=START_YEAR,
            start_month=START_MONTH,
            end_year=END_YEAR,
            end_month=END_MONTH,
            data_dir=DATA_DIR,
            anonymizer=anonymizer
        )

