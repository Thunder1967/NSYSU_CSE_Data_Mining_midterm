import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import optuna
import seaborn as sns
import matplotlib.pyplot as plt


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# 3-8 mapping 到 0-5
quality_mapping = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5}
quality_inv_mapping = {v:k for k,v in quality_mapping.items()}
train_data['quality'] = train_data['quality'].map(quality_mapping)
test_data['quality'] = test_data['quality'].map(quality_mapping)

# data preprocessing

train_data['alcohol_vs_acid'] = train_data['alcohol'] / train_data['volatile acidity']
test_data['alcohol_vs_acid'] = test_data['alcohol'] / test_data['volatile acidity']
drop_features = ['Id','pH','residual sugar', 'free sulfur dioxide']
# drop_features = ['Id']

x_train = train_data.drop(['quality'] + drop_features, axis=1)
y_train = train_data['quality']
x_test = test_data.drop(['quality'] + drop_features, axis=1)
y_test = test_data['quality']

cat_features = list(x_train.select_dtypes(include=['object', 'category']).columns)

# 定義 Optuna 目標函數
def objective(trial):
    params = {
        "iterations": 1000,  
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0.1, 1.0),
        "od_type": "Iter",
        "od_wait": 50,
        "verbose": False,
        "allow_writing_files": False,
        "random_seed": 42
    }
    
    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train, cat_features=cat_features)
    
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    return acc



# 調參
# print("開始優化超參數...")
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# print("\nBest parameters:", study.best_params)

# 使用最佳參數訓練最終模型
# best_params = study.best_params
best_params = {
    'learning_rate': 0.0022430210754648392, 
    'depth': 9, 
    'l2_leaf_reg': 6.415154438179145, 
    'random_strength': 0.9916513276325815
}

# 模型設定
model = CatBoostClassifier(
    iterations=5000,                      
    loss_function='MultiClassOneVsAll', 
    **best_params,
    early_stopping_rounds=100, 
    verbose=100            
)

# 訓練模型
model.fit(
    x_train, y_train,
    cat_features=cat_features,
)

feature_importance = model.get_feature_importance()
feature_names = x_train.columns

# 計算並輸出 Accuracy
train_preds = model.predict(x_train)
test_preds = model.predict(x_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print("-" * 30)
print(f"訓練集準確率 (Train Accuracy): {train_acc:.2%}")
print(f"測試集準確率 (Test Accuracy): {test_acc:.2%}")
print("-" * 30)



# 畫圖
sns.boxplot(x='quality', y='alcohol', data=train_data)
y_pred = model.predict(x_test).flatten()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.show()