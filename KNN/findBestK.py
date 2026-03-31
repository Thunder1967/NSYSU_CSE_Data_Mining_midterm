import myKNN
import myUtil
import myPreprocess
import numpy as np
import matplotlib.pyplot as plt

res = []
KNN1 = myKNN.BruteKNN("train.csv",myPreprocess.Scale_IQR_Preprocess(),myUtil.euclidean_distance_sq,0)
X_test,Y_test = myUtil.read_data("test.csv")
X_test = KNN1.preprocess.testPreprocess(X_test)
for i in range(2,100):
    KNN1.setK(i)
    res.append((i,KNN1.getTrainingAccuracy(),KNN1.getAccuracy(X_test,Y_test)))

# following code made by gemini
# 1. 提取資料
k_values = [item[0] for item in res]
train_accs = [item[1] for item in res]
test_accs = [item[2] for item in res]

# 2. 找出最大值及其對應的 k (使用前面學過的 np.argmax)
max_train_idx = np.argmax(train_accs)
max_train_k = k_values[max_train_idx]
max_train_val = train_accs[max_train_idx]

max_test_idx = np.argmax(test_accs)
max_test_k = k_values[max_test_idx]
max_test_val = test_accs[max_test_idx]

# 3. 建立畫布與繪圖
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accs, label='Training Accuracy', color='blue', alpha=0.6, linewidth=2)
plt.plot(k_values, test_accs, label='Testing Accuracy', color='red', alpha=0.6, linewidth=2)

# 4. 標註最大值 (使用 scatter 畫點，annotate 寫文字)
# 標註訓練集最大值
plt.scatter(max_train_k, max_train_val, color='darkblue', s=60, edgecolors='white', zorder=5)
plt.annotate(f'Max Train: {max_train_val:.4f} (k={max_train_k})', 
             xy=(max_train_k, max_train_val), 
             xytext=(max_train_k + 5, max_train_val),
             arrowprops=dict(arrowstyle='->', color='darkblue'),
             fontsize=10, color='darkblue')

# 標註測試集最大值
plt.scatter(max_test_k, max_test_val, color='darkred', s=60, edgecolors='white', zorder=5)
plt.annotate(f'Max Test: {max_test_val:.4f} (k={max_test_k})', 
             xy=(max_test_k, max_test_val), 
             xytext=(max_test_k + 5, max_test_val - 0.05),
             arrowprops=dict(arrowstyle='->', color='darkred'),
             fontsize=10, color='darkred')

# 5. 圖表修飾
plt.title('KNN Accuracy vs. $k$ Value', fontsize=14)
plt.xlabel('$k$ (Number of Neighbors)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

# 6. 儲存圖片
plt.savefig('KNN\\picture\\knn_comparison.png', bbox_inches='tight')