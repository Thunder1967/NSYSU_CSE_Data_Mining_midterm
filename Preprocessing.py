import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def Preprocessing():
    #讀取資料
    #df = pd.read_csv("WineQT.csv")
    train_df = pd.read_csv("train.csv")
    feature = train_df.drop(columns=['quality', 'Id'])
    print(len(train_df))

    #標準化
    df_zscore = (feature - feature.mean()) / feature.std()

    #剔除離群值
    train_clean = train_df[(df_zscore.abs() <= 3).all(axis=1)]
    print(len(train_clean))

    #資料重新輸出（記得取消註解）
    #train_clean.to_csv('pre_train.csv', index=False)

    
    #數據降維（主成分分析）
    train_data_without_Id = train_clean.drop(columns=['Id'])
    train_feature = train_clean.drop(columns=['quality', 'Id'])
    quality = train_clean[['quality']]
    Id = train_clean[['Id']]
    #數據降維（主成分分析）
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(train_feature)

    pca = PCA(n_components=8) #維度數量
    df_transformed = pca.fit_transform(data_scaled)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained variance ratio:", pca.explained_variance_ratio_.sum())

    #各變數對主成分貢獻度
    pca_columns = [f'PC{i+1}' for i in range(8)]
    components_df = pd.DataFrame(pca.components_, columns=train_feature.columns, index=pca_columns)
    components_df.to_csv("pca_feature.csv", encoding='utf-8-sig')

    #資料降維
    df_pca = pd.DataFrame(df_transformed, columns=pca_columns)
    df_pca = pd.concat([df_pca, quality.reset_index(drop=True)], axis=1)
    df_pca = pd.concat([df_pca, Id.reset_index(drop=True)], axis=1)
    df_pca.to_csv("pre_train_pca.csv", index=False, encoding='utf-8-sig')
    print(df_pca.head())

if __name__ == '__main__':
    Preprocessing()