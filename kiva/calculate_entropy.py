import numpy as np
import pandas as pd
from scipy.stats import entropy

dataset_users = pd.read_csv('/data/rating_5000_cls_hc_10core_iterative.csv')
dataset_features = pd.read_csv('/data/loan_feature_df_hc_10core.csv')

#set header names
dataset_users.columns = ['user_id', 'loan_id', 'rating']
dataset_features.columns = ['loan_id', 'feature_id', 'feature_value']

#merge the two datasets on loan_id
dataset_merged = pd.merge(dataset_users, dataset_features, on='loan_id')

unique_features = dataset_merged['feature_id'].unique()

n_unique_features = len(unique_features)

if n_unique_features <= 1:
    dataset_merged['entropy'] = 0
else:
    #create a pivot table
    pivot_table = pd.pivot_table(dataset_merged, values='feature_value', index=['user_id', 'loan_id', 'rating'], columns=['feature_id'], aggfunc=np.sum)
    pivot_table = pivot_table.fillna(0)
    pivot_table = pivot_table.applymap(lambda x: 1 if x > 0 else 0)
    #calculate entropy
    dataset_merged['entropy'] = pivot_table.apply(lambda x: entropy(x), axis=1)


