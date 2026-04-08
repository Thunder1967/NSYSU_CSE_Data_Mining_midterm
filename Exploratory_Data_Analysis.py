import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("WineQT.csv")
df_without_Id = df.drop(columns=['Id'])
df_feature = df.drop(columns=['quality', 'Id'])
quality = df[['quality']]
Id = df[['Id']]

print(df.shape) #資料結構
print(df.info()) #資料型態、缺失值

summary_table = df_without_Id.describe() #資料特性
summary_table.to_csv("data_description.csv", index=True, encoding='utf-8-sig')

#相關係數分析（皮爾森積差相關分析）
corr_matrix = df_without_Id.corr(method='pearson', numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Factors Correlation")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.show()


'''
如果要使用降維後的資料進行模型訓練
請複製到自己的演算法並修改變數，然後自行研究如何套用到測試
這裡只做為資料分析所使用
'''

#數據降維（主成分分析）
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_feature)

pca = PCA(n_components=5) #維度數量
df_transformed = pca.fit_transform(data_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance ratio:", pca.explained_variance_ratio_.sum())

#各變數對主成分貢獻度
pca_columns = [f'PC{i+1}' for i in range(5)]
components_df = pd.DataFrame(pca.components_, columns=df_feature.columns, index=pca_columns)
components_df.to_csv("pca_feature.csv", encoding='utf-8-sig')

'''
降維資料
df_pca = pd.DataFrame(df_transformed, columns=pca_columns)
df_pca = pd.concat([df_pca, quality.reset_index(drop=True)], axis=1)
df_pca = pd.concat([df_pca, Id.reset_index(drop=True)], axis=1)
df_pca.to_csv("WineQT_pca.csv", index=False, encoding='utf-8-sig')
print(df_pca.head())
'''