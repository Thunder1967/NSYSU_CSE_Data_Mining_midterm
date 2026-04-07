import pandas as pd
import os

# 讀取 train.csv
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, '..', 'train.csv')

# 讀取資料
df_train = pd.read_csv(file_path)
# print(df.head())

# 建立 X 與 y 
X_train = df_train.drop(['quality', 'Id'], axis = 1)
y_train = df_train['quality']

# 丟進模型訓練
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100) # 越多棵樹越準確
clf.fit(X_train, y_train)

# 讀取 test.csv
test_path = os.path.join(current_dir, '..', 'test.csv')
df_test = pd.read_csv(test_path)

# 準備 X 跟 y
X_test = df_test.drop(['quality', 'Id'], axis = 1)
y_test = df_test['quality']

# 算出 Accuracy
accuracy = clf.score(X_test, y_test)
print(f"這 100 棵樹投票出來的準確率是：{accuracy * 100:.2f}%")