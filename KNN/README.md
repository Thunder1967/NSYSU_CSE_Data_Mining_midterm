## 如何使用
### 建立模型
1. 環境: 準備 Python, numpy, pandas 並 import 檔案
    ```python
    import myKNN
    import myUtil
    import myPreprocess
    ```
3. 讀入資料 `X_train,Y_train = myUtil.read_data("train.csv")`
4. 選擇模型演算法 `myKNN.BruteKNN` 或 `myKNN.BallTreeKNN` ，通常是 Brute 效能較佳
5. 選擇預處理方式，從 `myPreprocess.py` 挑一個 class
    - STD : 標準化
    - Scale : 正規化
    - IQRR : 刪除離群值
    - IQRC : 蓋帽法
6. 選擇距離計算方式 
    1. `myUtil.manhattan_distance()`
    2. `myUtil.euclidean_distance()`
    3. `myUtil.euclidean_distance_sq()`
       -  `myKNN.BallTreeKNN` 無法使用 `myUtil.euclidean_distance_sq()`
7. 決定 K
8. 建立模型
    ```python
    KNN_model = BruteKNN(X_train,Y_train,Preprocess,distance_fnc,K)
    # example
    KNN_model = BruteKNN(X_train,Y_train,myPreprocess.STD_Preprocess(),myUtil.euclidean_distance_sq,10)
    ```
### 模型評估
#### training accuracy
```python 
KNN_model.getTrainingAccuracy()
```
#### testing accuracy
```python 
X_test,Y_test = myUtil.read_data("test.csv")
KNN_model.getTestingAccuracy(X_test,Y_test)
```
## 程式架構
1. `myKNN.py`
   - 實現 KNN 主要演算法
2. `myPreprocess.py`
   - 實現各種預處理方法
3. `myUtil.py`
   - 距離公式,讀檔,程式時間檢測器,IQR計算,Kfold實作
4. `made_by_AI.py`
   - 繪製 K 對 testing 和 training accuracy 的折線圖
   - made_by_AI.py 的程式由 Gemini 生成
5. `testK.py` & `testK_Kfold.py`
   - 測試測試模型能力，選擇最佳預處理方法和 K 值
