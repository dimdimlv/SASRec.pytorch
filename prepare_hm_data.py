# prepare_hm_data.py
import pandas as pd
from collections import defaultdict
import numpy as np


def prepare_data(input_path='./data/HM/transactions.csv', output_path='./data/interim/hm_data.pkl', sample_size=None, random_seed=42):
    # Load the transaction data
    transactions = pd.read_csv(input_path)

    # Check if sampling is required
    if sample_size is not None:
        # Get a random sample of user IDs
        unique_users = transactions['customer_id'].unique()
        np.random.seed(random_seed)
        sampled_users = np.random.choice(unique_users, size=sample_size, replace=False)

        # Filter transactions to include only sampled users
        transactions = transactions[transactions['customer_id'].isin(sampled_users)]
        print(f"Sampled {sample_size} users with {len(transactions)} total interactions.")

    # Keep only necessary columns and sort by time
    interactions = transactions[['customer_id', 'article_id', 't_dat']]
    interactions = interactions.rename(
        columns={'customer_id': 'user_id', 'article_id': 'item_id', 't_dat': 'timestamp'})
    interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])

    # Sort interactions by user and timestamp
    interactions = interactions.sort_values(by=['user_id', 'timestamp'])

    # Group by user and create train/valid/test splits
    user_groups = interactions.groupby('user_id')
    user_train, user_valid, user_test = defaultdict(list), defaultdict(list), defaultdict(list)

    for user_id, group in user_groups:
        item_sequence = group['item_id'].tolist()
        if len(item_sequence) < 3:  # Not enough data for train/valid/test
            user_train[user_id] = item_sequence
            user_valid[user_id] = []
            user_test[user_id] = []
        else:
            user_train[user_id] = item_sequence[:-2]
            user_valid[user_id] = [item_sequence[-2]]
            user_test[user_id] = [item_sequence[-1]]

    # Save the data as a pickle file for easy loading
    data = {'user_train': user_train, 'user_valid': user_valid, 'user_test': user_test}
    pd.to_pickle(data, output_path)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare H&M dataset for SASRec")
    parser.add_argument('--input_path', type=str, default='transactions.csv', help='Path to transactions CSV file')
    parser.add_argument('--output_path', type=str, default='hm_data.pkl', help='Output path for processed data')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of users to sample')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for sampling')

    args = parser.parse_args()

    prepare_data(input_path=args.input_path, output_path=args.output_path, sample_size=args.sample_size,
                 random_seed=args.random_seed)