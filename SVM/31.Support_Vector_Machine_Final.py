import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

train_data = pd.read_csv("pre_train.csv")
test_data = pd.read_csv("test.csv")

train_data_feature = train_data.drop(columns=['quality'])
train_data_quality = train_data['quality']
test_feature = test_data.drop(columns=['quality', 'Id'])
test_Id = test_data['Id']

models = {
    "linear": SVC(kernel='linear', C=5.1, decision_function_shape='ovr'),
    "poly": SVC(kernel='poly', C=0.25, degree=3, gamma=0.2, decision_function_shape='ovr'),
    "rbf": SVC(kernel='rbf', C=3.7, gamma=0.8, decision_function_shape='ovr')
}
    
for name, svm_model in models.items():

    pipe = Pipeline([
        ('std_scalar', StandardScaler()),
        ('svm', svm_model)
    ])
    pipe.fit(train_data_feature, train_data_quality)

    train_data_quality_pred = pipe.predict(train_data_feature)
    train_f1 = f1_score(train_data_quality, train_data_quality_pred, average='weighted')
    print(f"{name.lower()} Train F1-Score: {train_f1:.4f}")

    preds = pipe.predict(test_feature)
    output = pd.DataFrame({'Id': test_Id, 'quality': preds})
    output.to_csv(f"final_pred_{name}.csv", index=False)
    
preds_linear = pd.read_csv("final_pred_linear.csv")
preds_poly = pd.read_csv("final_pred_poly.csv")
preds_rbf = pd.read_csv("final_pred_rbf.csv")
p1 = preds_linear['quality']
p2 = preds_poly['quality']
p3 = preds_rbf['quality']
avg_preds = ((p1 + p2 + p3) / 3).round().astype(int)

#混和結果
preds_blend = pd.DataFrame({'Id': test_data['Id'], 'quality': avg_preds})
preds_blend.to_csv(f"final_pred_blend.csv", index=False)