import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#資料讀取
df = pd.read_csv("WineQT.csv")
df_without_Id = df.drop(columns=['Id'])
df_feature = df.drop(columns=['quality', 'Id'])
df_quality = df[['quality']]
df_Id = df[['Id']]

print(df.shape) #資料結構
print(df.info()) #資料型態、缺失值

description_table = df_without_Id.describe().transpose() #資料特性
df_skew = df_feature.skew() #偏度
df_kurt = df_feature.kurt() #峰度
description_table['skewness'] = df_skew
description_table['kurtosis'] = df_kurt
description_table.to_csv("data_description.csv", index=True) #生成檔案

#皮爾森積差相關分析
corr_matrix = df_without_Id.corr(method='pearson', numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.show()


'''
This is only for EDA
If you want to use PCA for model training, do not just copy and paste
'''

#主成分分析
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_feature)

pca = PCA(n_components=5) #維度數量
df_transformed = pca.fit_transform(data_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Ratio sum:", pca.explained_variance_ratio_.sum())

#各變數對主成分貢獻度
pca_columns = [f'PC{i+1}' for i in range(5)]
df_components = pd.DataFrame(pca.components_, columns=df_feature.columns, index=pca_columns)
df_components.to_csv("pca_components.csv", index=False)

'''
#特徵降維
df_pca = pd.DataFrame(df_transformed, columns=pca_columns)
df_pca['df_quality'] = df_quality
df_pca['Id'] = df_Id
df_pca.to_csv("WineQT_pca.csv", index=False)
print(df_pca.head())
'''
