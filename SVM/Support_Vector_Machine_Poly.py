import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def SupportVectorMachine_Poly():

    train_data = pd.read_csv("pre_train.csv")
    X = train_data.drop(columns=['quality', 'Id'])
    y = train_data[['quality']]

    # K-fold
    skf = StratifiedKFold(n_splits=10)

    pipe = Pipeline([
                ('std_scalar', StandardScaler()),
                ('svm', SVC(kernel='poly', decision_function_shape='ovr'))
    ])

    param_grid = {
        'svm__C': [0.1, 0.5, 1],
        'svm__degree': [3, 5],
        'svm__gamma': [0.15, 0.2, 0.25, 0.3]
    }
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring='f1_weighted', return_train_score=True, n_jobs=-1, verbose=2)
    grid.fit(X, y.values.ravel())

    results = pd.DataFrame(grid.cv_results_)
    output_columns = {
        'param_svm__C': 'C_Parameter',
        'param_svm__degree': 'Degree_Parameter',
        'param_svm__gamma': 'Gamma_Parameter',
        'mean_train_score': 'Mean_F1_Score(Train)',
        'std_train_score': 'Std_F1_Score(Train)',
        'mean_test_score': 'Mean_F1_Score(Test)',
        'std_test_score': 'Std_F1_Score(Test)',
        'rank_test_score': 'Rank'
    }
    table = results[list(output_columns.keys())].rename(columns=output_columns)
    table = table.sort_values(by='Rank')

    table.to_csv("results_poly.csv", index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    SupportVectorMachine_Poly()