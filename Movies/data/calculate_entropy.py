import numpy as np
import pandas as pd
from scipy.stats import entropy

dataset_users = pd.read_csv('data/rating_5000_cls_hc_10core_iterative.csv', delimiter='\t', header=None)
dataset_features = pd.read_csv('data/loan_feature_df_hc_10core.csv',header=None)

# Set header names
dataset_users.columns = ['user_id', 'loan_id', 'rating']
dataset_features.columns = ['loan_id', 'feature_id', 'feature_value']

# Merge the two datasets on loan_id
dataset_merged = pd.merge(dataset_users, dataset_features, on='loan_id')

unique_features = dataset_merged['feature_id'].unique()

# Initialize an empty list to store the results
results = []

for user in dataset_merged['user_id'].unique():
    user_data = dataset_merged[dataset_merged['user_id'] == user]
    count_loans_funded = np.zeros(len(unique_features))
    
    for idx, feature in enumerate(unique_features):
        count_loans_with_feature = len(user_data[user_data['feature_id'] == feature])
        count_loans_funded[:count_loans_with_feature] = 1
        
        entropy_val = entropy(count_loans_funded)
        
        results.append([user, feature, entropy_val])

# Create a new DataFrame from the results list
results_df = pd.DataFrame(results, columns=['user_id', 'feature_id', 'entropy'])

print(results_df.head())
