import pandas as pd

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

if __name__ == '__main__':
    Preprocessing()