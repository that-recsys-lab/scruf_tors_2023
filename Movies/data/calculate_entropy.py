import numpy as np
import pandas as pd
from scipy.stats import entropy

dataset_users = pd.read_csv('./ratings.csv', header=None)
dataset_features = pd.read_csv('./items.csv',header=None)

# Set header names
dataset_users.columns = ['user_id', 'movie_id', 'rating']
dataset_features.columns = ['movie_id', 'feature_id', 'feature_value']

# Merge the two datasets on movie_id
dataset_merged = pd.merge(dataset_users, dataset_features, on='movie_id')

unique_features = dataset_merged['feature_id'].unique()

# Initialize an empty list to store the results
results = []

for user in dataset_merged['user_id'].unique():
    user_data = dataset_merged[dataset_merged['user_id'] == user]
    count_movies_funded = np.zeros(len(unique_features))
    
    for idx, feature in enumerate(unique_features):
        count_movies_with_feature = len(user_data[user_data['feature_id'] == feature])
        count_movies_funded[:count_movies_with_feature] = 1
        count_movies[:len(count_movies)+1] = 1
        
        entropy_val = entropy(count_movies_funded)
        
        results.append([user, feature, entropy_val])

# Create a new DataFrame from the results list
results_df = pd.DataFrame(results, columns=['user_id', 'feature_id', 'entropy'])

print(results_df.head())
