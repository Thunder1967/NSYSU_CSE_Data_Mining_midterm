import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

x_train = train_df.drop(['quality', 'Id'], axis=1)
y_train = train_df['quality']

x_test = test_df.drop(['quality', 'Id'], axis=1)
y_test = test_df['quality']

y_train_shifted = y_train - y_train.min()
y_test_shifted = y_test - y_test.min()

def score(m, x_tr, y_tr, x_te, y_te, train=True):
    curr_x, curr_y = (x_tr, y_tr) if train else (x_te, y_te)
    title = 'Train Result' if train else 'Test Result'
    
    pred = m.predict(curr_x)
    print(f'--- {title} ---')
    print(f"Accuracy Score: {accuracy_score(curr_y, pred)*100:.2f}%")    
    print(f"Precision Score: {precision_score(curr_y, pred, average='weighted')*100:.2f}%")
    print(f"Recall Score: {recall_score(curr_y, pred, average='weighted')*100:.2f}%")
    print(f"F1 score: {f1_score(curr_y, pred, average='weighted')*100:.2f}%")
    print(f"Confusion Matrix:\n {confusion_matrix(curr_y, pred)}")


n_estimators = [int(x) for x in np.linspace(start=50, stop = 200, num = 151)]
max_depth = [int(x) for x in np.linspace(3, 5, num=3)]
learning_rate = 0.2
colsample_bytree =[round(float(x),2) for x in np.linspace(start=0.01, stop=0.4, num=100)]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'colsample_bytree': colsample_bytree}
random_grid

xg4 = xgb.XGBClassifier(learning_rate = 0.2, random_state=330)

#Random search of parameters, using 3 fold cross validation, search across 100 different combinations, and use all available cores
xg_random = RandomizedSearchCV(estimator = xg4, param_distributions=random_grid,
                              n_iter=100, cv=3, verbose=2, random_state=330, n_jobs=-1)

print("Starting Parameter Search...")
xg_random.fit(x_train, y_train_shifted)

best_p = xg_random.best_params_
print("\n" + "="*45)
print("【XGBoost Hyperparameter Tuning Results】")
print(f"Best n_estimators: {best_p.get('n_estimators')}")
print(f"Best max_depth: {best_p.get('max_depth')}")
print(f"Best colsample_bytree: {best_p.get('colsample_bytree')}")
print(f"Fixed learning_rate: 0.2")
print("="*45)
xg5 = xgb.XGBClassifier(**best_p, learning_rate = 0.2, random_state = 330)
xg5.fit(x_train, y_train_shifted)

score(xg5, x_train, y_train_shifted, x_test, y_test_shifted, train=True)
print("\n" + "-"*35)
score(xg5, x_train, y_train_shifted, x_test, y_test_shifted, train=False)