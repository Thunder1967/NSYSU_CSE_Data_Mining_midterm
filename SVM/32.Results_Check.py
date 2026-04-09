import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import seaborn as sns

#資料讀取
linear_pred = pd.read_csv("final_pred_linear.csv")
poly_pred = pd.read_csv("final_pred_poly.csv")
rbf_pred = pd.read_csv("final_pred_rbf.csv")
blend_pred = pd.read_csv("final_pred_blend.csv")
test_data = pd.read_csv("test.csv")
test_answer = test_data[['Id', 'quality']]

#資料整理
linear_pred = linear_pred.sort_values(by='Id', ascending=True, na_position='last')
poly_pred = poly_pred.sort_values(by='Id', ascending=True, na_position='last')
rbf_pred = rbf_pred.sort_values(by='Id', ascending=True, na_position='last')
blend_pred = blend_pred.sort_values(by='Id', ascending=True, na_position='last')
test_answer = test_answer.sort_values(by='Id', ascending=True, na_position='last')

linear_pred_quality = linear_pred['quality']
poly_pred_quality = poly_pred['quality']
rbf_pred_quality = rbf_pred['quality']
blend_pred_quality = blend_pred['quality']
test_answer_quality = test_answer['quality']
labels = [3, 4, 5, 6, 7, 8]

#混淆矩陣
cm_linear = confusion_matrix(test_answer_quality, linear_pred_quality, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=100)
plt.xlabel('Predicted Quality')
plt.ylabel('True Quality')
plt.title('Linear SVM Confusion Matrix')
#plt.show()

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