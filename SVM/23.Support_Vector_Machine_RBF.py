import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv("pre_train.csv")
train_data_feature = train_data.drop(columns=['quality'])
train_data_quality = train_data['quality']

# K-fold
skf = StratifiedKFold(n_splits=10)

# SVM_RBF
pipe = Pipeline([
            ('std_scalar', StandardScaler()),
            ('svm', SVC(kernel='rbf', decision_function_shape='ovr'))
])

param_grid = {
    'svm__C': [3.5, 3.7, 3.8, 3.9, 4, 4.1, 4.2],
    'svm__gamma': [0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 1]
}
grid = GridSearchCV(pipe, param_grid, cv=skf, scoring='f1_weighted', return_train_score=True, n_jobs=-1, verbose=2)
grid.fit(train_data_feature, train_data_quality)

results = pd.DataFrame(grid.cv_results_)
output_columns = {
    'param_svm__C': 'C_Parameter',
    'param_svm__gamma': 'Gamma_Parameter',
    'mean_train_score': 'Mean_F1_Score (Train)',
    'std_train_score': 'Std_F1_Score (Train)',
    'mean_test_score': 'Mean_F1_Score (Test)',
    'std_test_score': 'Std_F1_Score (Test)',
    'rank_test_score': 'Rank'
}
table = results[list(output_columns.keys())].rename(columns=output_columns)
table = table.sort_values(by='Rank')

table.to_csv("results_rbf.csv", index=False)