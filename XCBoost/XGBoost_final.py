import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

# 1. 資料讀取與預處理
# 讀取訓練與測試集資料
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 特徵工程：移除目標變數 'quality' 與不具預測意義的標記 'Id'
x_train = train_df.drop(['quality', 'Id'], axis=1)
y_train = train_df['quality']
x_test = test_df.drop(['quality', 'Id'], axis=1)
y_test = test_df['quality']

# XGBoost 分類標籤轉換
# XGBoost 的分類器要求標籤必須從 0 開始 (例如：若品質是 3~8，需轉為 0~5)
y_min = y_train.min()
y_train_shifted = y_train - y_min
y_test_shifted = y_test - y_min

# 2. 定義評估函數 (Evaluation Function)
def score(m, x_tr, y_tr, x_te, y_te, train=True):
    """
    用於輸出模型的各項評估指標
    m: 訓練好的模型
    train: True 表示輸出訓練集結果，False 表示輸出測試集結果
    """
    curr_x, curr_y = (x_tr, y_tr) if train else (x_te, y_te)
    title = '【Train Result】' if train else '【Test Result】'
    
    # 進行預測
    pred = m.predict(curr_x)
    
    print(f'\n{title}')
    # 準確率：預測正確的比例
    print(f"Accuracy Score: {accuracy_score(curr_y, pred)*100:.2f}%")    
    # 精確率：考量類別不平衡，使用 'weighted' 平均
    print(f"Precision Score: {precision_score(curr_y, pred, average='weighted', zero_division=0)*100:.2f}%")
    # 召回率：模型捕捉正樣本的能力
    print(f"Recall Score: {recall_score(curr_y, pred, average='weighted')*100:.2f}%")
    # F1 Score：精確率與召回率的調和平均，綜合評估指標
    print(f"F1 score: {f1_score(curr_y, pred, average='weighted')*100:.2f}%")
    # 混淆矩陣：查看具體的誤判分佈
    print(f"Confusion Matrix:\n {confusion_matrix(curr_y, pred)}")

# 3. 定義超參數搜尋空間 (Search Space)
# 這裡設定的範圍偏向「強正則化」，目的是防止模型過度擬合訓練集
n_estimators = [int(x) for x in np.linspace(start=50, stop=200, num=151)] # 樹的數量
max_depth = [2, 3, 4, 5] # 限制樹深，深度越小模型越簡單，越不容易過擬合
colsample_bytree = [round(float(x), 2) for x in np.linspace(start=0.01, stop=0.6, num=100)] # 每棵樹隨機採樣特徵比例
min_child_weight = [10, 15, 20] # 節點劃分所需的最小權重和，數值越大越保守
subsample = [0.7, 0.8, 0.9] # 每棵樹隨機採樣樣本比例
reg_lambda = [int(x) for x in np.linspace(start=100, stop=1000, num=100)] # L2 正則化參數，越高則懲罰越重

random_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'colsample_bytree': colsample_bytree,
    'min_child_weight': min_child_weight,
    'subsample': subsample,
    'reg_lambda': reg_lambda
}

# 4. 執行隨機搜尋 (Randomized Search)
# 建立基礎分類器，learning_rate 設為 0.2 以兼顧速度與收斂
xg_base = xgb.XGBClassifier(learning_rate=0.2, random_state=330)

xg_random = RandomizedSearchCV(
    estimator=xg_base, 
    param_distributions=random_grid,
    n_iter=100,           # 隨機挑選 100 組參數嘗試
    cv=3,                 # 3 折交叉驗證
    verbose=1,            # 顯示搜尋進度
    random_state=330,     # 固定隨機種子以利復現結果
    n_jobs=-1,            # 使用所有 CPU 核心加速運算
    return_train_score=True # 關鍵：必須回傳訓練集分數以便後續計算 Gap
)

#當你設定 cv=3 時，電腦會自動幫你執行以下循環：
#第一步：切分資料
#將你的 訓練資料 (Train Data) 隨機打亂並平均分成 3 份（三個摺疊，Folds）。我們稱之為 A、B、C。
#第二步：執行三次實驗
#第 1 次實驗： 用 B + C 訓練模型，用 A 進行驗證（得到分數 1）。
#第 2 次實驗： 用 A + C 訓練模型，用 B 進行驗證（得到分數 2）。
#第 3 次實驗： 用 A + B 訓練模型，用 C 進行驗證（得到分數 3）。
#第三步：計算平均
#將這三次的分數取 平均值，這就是 RandomizedSearchCV 用來判斷這組參數好壞的最終依據。

print(">>> 執行「絕對穩定優先」搜尋 (Priority: Highest Balanced Score)...")
xg_random.fit(x_train, y_train_shifted)

# 5. 核心邏輯：穩定性篩選 (Stability Filtering)
# 提取所有嘗試過的參數組合結果
results = pd.DataFrame(xg_random.cv_results_)

# 計算「推廣誤差」(Gap)：訓練集準確度 - 驗證集準確度
# Gap 越大，表示模型越容易在現實數據上失效 (Overfitting)
results['gap'] = results['mean_train_score'] - results['mean_test_score']

# --- 定義平衡分數 (Balanced Score) ---
# 公式：驗證集得分 - (10.0 * Gap)
# 這裡使用 10 倍的高額懲罰，強制排除任何「在訓練集表現極好、但測試集掉漆」的參數
results['balanced_score'] = results['mean_test_score'] - (10.0 * results['gap'])

# 依照平衡分數降序排列，取分數最高的一組 (即：預估最穩定的參數)
best_balanced_row = results.sort_values('balanced_score', ascending=False).iloc[0]
final_params = best_balanced_row['params']

print("\n" + "="*50)
print("【XGBoost 穩定性優化成果】")
print(f"獲選最佳平衡參數：\n{final_params}")
print(f"\nCV 交叉驗證預估指標：")
print(f" - 預計 Test Accuracy: {best_balanced_row['mean_test_score']*100:.2f}%")
print(f" - 預計 Gap (Train-Test): {best_balanced_row['gap']*100:.2f}%")
print(f" - 最終平衡得分 (含懲罰): {best_balanced_row['balanced_score']:.4f}")
print("="*50)

# 6. 最終模型訓練與驗證
# 使用篩選出的最佳參數重新訓練完整模型
final_model = xgb.XGBClassifier(**final_params, learning_rate=0.2, random_state=330)
final_model.fit(x_train, y_train_shifted)

# 輸出訓練集與測試集的最終數據
score(final_model, x_train, y_train_shifted, x_test, y_test_shifted, train=True)
print("-" * 30)
score(final_model, x_train, y_train_shifted, x_test, y_test_shifted, train=False)