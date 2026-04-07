import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def SupportVectorMachine():

    train_data = pd.read_csv("pre_train.csv")
    test_data = pd.read_csv("test.csv")
    X_train = train_data.drop(columns=['quality', 'Id'])
    y_train = train_data[['quality']]
    X_test = test_data.drop(columns=['quality', 'Id'])

    models = {
        "linear": SVC(kernel='linear', C=5.1),
        "poly": SVC(kernel='poly', C=0.1, degree=3, gamma=0.3),
        "rbf": SVC(kernel='rbf', C=2.2, gamma=0.3)
    }
    
    for name, svm_model in models.items():

        pipe = Pipeline([
            ('std_scalar', StandardScaler()),
            ('svm', svm_model)
        ])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        output = pd.DataFrame({'Id': test_data['Id'], 'quality': preds})
        output.to_csv(f"final_pred_{name}.csv", index=False)

if __name__ == '__main__':
    SupportVectorMachine()