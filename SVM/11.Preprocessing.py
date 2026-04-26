import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#資料讀取
train_df = pd.read_csv("train.csv")
#train_df = pd.read_csv("train_augmentation.csv")
train_df_feature = train_df.drop(columns=['quality', 'Id'])
print(len(train_df))

#Z-score
train_df_zscore = (train_df_feature - train_df_feature.mean()) / train_df_feature.std()
train_clean = train_df[(train_df_zscore.abs() <= 3).all(axis=1)]
print(len(train_clean))

train_clean = train_clean.drop(columns=['Id'])

#資料匯出
train_clean.to_csv('pre_train.csv', index=False)

'''
#主成分分析
train_clean_feature = train_clean.drop(columns=['quality'])
train_clean_quality = train_clean['quality']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(train_clean_feature)

pca = PCA(n_components=5) #維度數量
df_transformed = pca.fit_transform(data_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Ratio sum:", pca.explained_variance_ratio_.sum())

#各變數對主成分貢獻度
pca_columns = [f'PC{i+1}' for i in range(5)]
df_components = pd.DataFrame(pca.components_, columns=train_clean_feature.columns, index=pca_columns)
df_components.to_csv("train_pca_components.csv", index=False)

#特徵降維
df_pca = pd.DataFrame(df_transformed, columns=pca_columns)
df_pca['df_quality'] = train_clean_quality
df_pca.to_csv("pre_train_pca.csv", index=False)
print(df_pca.head())
'''