import praw
import pandas as pd

# Reddit API credentials
REDDIT_CLIENT_ID = "Your_reddit_client_id"
REDDIT_CLIENT_SECRET = "Your_reddit_client_secret"
REDDIT_USER_AGENT = "python:stock_analysis_project:v1.0.0 (by /u/Your_reddit_username)"

def fetch_reddit_data(subreddit_name, query, limit=500):
    """
    Fetch posts from a specific subreddit based on a query.

    :param subreddit_name: The subreddit to search (e.g., 'stocks').
    :param query: The search query (e.g., 'stock market').
    :param limit: The number of posts to fetch.
    :return: DataFrame containing post details.
    """
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.search(query, limit=limit)

    data = []
    for post in posts:
        data.append({
            'title': post.title,
            'selftext': post.selftext,
            'created_utc': post.created_utc,
            'score': post.score,
            'num_comments': post.num_comments,
        })

    return pd.DataFrame(data)

if __name__ == "__main__":






