'''
SMOTE has a lots of variant
Change the "import" and "oversample" to find which fit the best
scource: https://www.geeksforgeeks.org/machine-learning/smote-for-imbalanced-classification-with-python/

Also, the "train_augmentation" file doesn't have "Id" column
'''

import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

#資料讀取
train_df = pd.read_csv("train.csv")
train_df_without_Id = train_df.drop(columns=['Id'])

# SMOTE 算法
oversample = BorderlineSMOTE() # Change the variant if needed
features, labels = oversample.fit_resample(train_df_without_Id.drop(columns=['quality']), train_df_without_Id['quality'])

#資料匯出
data_augmentation = features.copy()
data_augmentation['quality'] = labels
data_augmentation.to_csv('train_augmentation.csv', index=False)