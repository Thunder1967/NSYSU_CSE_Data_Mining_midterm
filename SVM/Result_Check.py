import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt 
import seaborn as sns

def Result():
    #讀取資料
    linear_df = pd.read_csv("final_pred_linear.csv")
    poly_df = pd.read_csv("final_pred_poly.csv")
    rbf_df = pd.read_csv("final_pred_rbf.csv")
    test_df = pd.read_csv("test.csv")
    test_df = test_df[['Id', 'quality']]

    #整理資料
    linear_df = linear_df.sort_values(by='Id', ascending=True, na_position='last').reset_index(drop=True)
    poly_df = poly_df.sort_values(by='Id', ascending=True, na_position='last').reset_index(drop=True)
    rbf_df = rbf_df.sort_values(by='Id', ascending=True, na_position='last').reset_index(drop=True)
    test_df = test_df.sort_values(by='Id', ascending=True, na_position='last').reset_index(drop=True)

    #準確率
    acc_linear = (linear_df['quality'] == test_df['quality']).mean()
    acc_poly = (poly_df['quality'] == test_df['quality']).mean()
    acc_rbf = (rbf_df['quality'] == test_df['quality']).mean()

    #fi score
    f1_linear = f1_score(test_df['quality'], linear_df['quality'], average='weighted')
    f1_poly = f1_score(test_df['quality'], poly_df['quality'], average='weighted')
    f1_rbf = f1_score(test_df['quality'], rbf_df['quality'], average='weighted')

    #輸出
    print(f"Linear Accuracy: {acc_linear:.2%}")
    print(f"Linear F1 Score: {f1_linear:.4f}")
    print(f"Poly Accuracy: {acc_poly:.2%}")
    print(f"Poly 模型的 F1 Score 為: {f1_poly:.4f}")
    print(f"RBF Accuracy: {acc_rbf:.2%}")
    print(f"RBF 模型的 F1 Score 為: {f1_rbf:.4f}")

    linear_pred = linear_df['quality']
    poly_pred = poly_df['quality']
    rbf_pred = rbf_df['quality']
    y_true = test_df['quality']
    labels = [3, 4, 5, 6, 7, 8]

    #混淆矩陣
    cm_linear = confusion_matrix(y_true, linear_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
    plt.xlabel('Predicted Quality')
    plt.ylabel('True Quality')
    plt.title('Linear SVM Confusion Matrix')
    plt.show()

    cm_poly = confusion_matrix(y_true, poly_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_poly, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
    plt.xlabel('Predicted Quality')
    plt.ylabel('True Quality')
    plt.title('Poly SVM Confusion Matrix')
    plt.show()

    cm_rbf = confusion_matrix(y_true, rbf_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
    plt.xlabel('Predicted Quality')
    plt.ylabel('True Quality')
    plt.title('RBF SVM Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    Result()