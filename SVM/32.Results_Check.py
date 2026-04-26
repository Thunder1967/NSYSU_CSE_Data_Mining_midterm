import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import seaborn as sns

def score(name, test_x, test_answer):
    test_x_quality = test_x['quality']
    test_answer_quality = test_answer['quality']
    
    print(f"▼//{name}Test Result")
    print(f"Accuracy Score: {accuracy_score(test_answer_quality, test_x_quality)*100:.2f}%")
    print(f"Precision Score: {precision_score(test_answer_quality, test_x_quality, zero_division=0, average='weighted')*100:.2f}%")
    print(f"Recall Score: {recall_score(test_answer_quality, test_x_quality, average='weighted')*100:.2f}%")
    print(f"F1 score: {f1_score(test_answer_quality, test_x_quality, average='weighted')*100:.2f}%")
    print(f"Confusion Matrix:\n {confusion_matrix(test_answer_quality, test_x_quality)}")
    print(f"")

#資料讀取
linear_pred = pd.read_csv("final_pred_linear.csv")
poly_pred = pd.read_csv("final_pred_poly.csv")
rbf_pred = pd.read_csv("final_pred_rbf.csv")
blend_pred = pd.read_csv("final_pred_blend.csv")
test_data = pd.read_csv("test.csv")
test_answer = test_data[['Id', 'quality']]

score("Linear", linear_pred, test_answer)
score("Poly", poly_pred, test_answer)
score("RBF", rbf_pred, test_answer)
score("Blend", blend_pred, test_answer)
"""
cm_poly = confusion_matrix(test_answer_quality, poly_pred_quality, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_poly, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.title('Poly SVM Confusion Matrix')
#plt.show()

cm_rbf = confusion_matrix(test_answer_quality, rbf_pred_quality, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.title('RBF SVM Confusion Matrix')
#plt.show()

cm_blend = confusion_matrix(test_answer_quality, blend_pred_quality, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_blend, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.title('Blend SVM Confusion Matrix')
#plt.show()

#評估輸出
report_linear = classification_report(test_answer_quality, linear_pred_quality, digits=4, zero_division=0)
print(f"linear Result: ")
print(report_linear)

report_poly = classification_report(test_answer_quality, poly_pred_quality, digits=4, zero_division=0)
print(f"poly Result: ")
print(report_poly)

report_rbf = classification_report(test_answer_quality, rbf_pred_quality, digits=4, zero_division=0)
print(f"rbf Result: ")
print(report_rbf)

report_blend = classification_report(test_answer_quality, blend_pred_quality, digits=4, zero_division=0)
print(f"blend Result: ")
print(report_blend)
"""