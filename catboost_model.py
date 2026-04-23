import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
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

# train_data['alcohol_vs_acid'] = train_data['alcohol'] / train_data['volatile acidity']
# test_data['alcohol_vs_acid'] = test_data['alcohol'] / test_data['volatile acidity']
drop_features = ['Id','pH','residual sugar', 'free sulfur dioxide']
# drop_features = ['Id']

x_train = train_data.drop(['quality'] + drop_features, axis=1)
y_train = train_data['quality']
x_test = test_data.drop(['quality'] + drop_features, axis=1)
y_test = test_data['quality']



cat_features = list(x_train.select_dtypes(include=['object', 'category']).columns)

# 模型設定
model = CatBoostClassifier(
    iterations=5000,        
    learning_rate=0.01,     
    depth=7,                
    loss_function='MultiClassOneVsAll', 
    random_strength=0.5,
    early_stopping_rounds=100, 
    verbose=100            
)



# 訓練模型
model.fit(
    x_train, y_train,
    cat_features=cat_features,
    eval_set=(x_test, y_test), 
)

feature_importance = model.get_feature_importance()
feature_names = x_train.columns

predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions.flatten())

print("-" * 30)
print(f"模型準確率 (Accuracy): {accuracy:.2%}")
print("-" * 30)



#畫圖
sns.boxplot(x='quality', y='alcohol', data=train_data)
y_pred = model.predict(x_test).flatten()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.show()