import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def count(fileName):
#     print(f'{fileName}:')
#     rData = pd.read_csv(fileName) # row data
#     rY = rData['quality'].to_numpy() # quality
#     val,cnt = np.unique(rY, return_counts=True)
    
#     # following code made by gemini
#     # 讀取資料
#     rData = pd.read_csv(fileName)
#     rY = rData['quality'].to_numpy()

#     # 計算唯一值與次數
#     val, cnt = np.unique(rY, return_counts=True)
    
#     # 建立圖表
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(val, cnt, color='steelblue', edgecolor='black', alpha=0.8)
    
#     # 設定圖表細節
#     plt.xlabel('Quality Score', fontsize=12)
#     plt.ylabel('Frequency (Count)', fontsize=12)
#     plt.title(f'Distribution of Quality - {fileName}', fontsize=14)
#     plt.xticks(val)  # 確保 X 軸只顯示出現過的品質分數
#     plt.grid(axis='y', linestyle='--', alpha=0.7) # 加入水平虛線輔助閱讀

#     # 在長條上方加上數值標籤
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                  f'{int(height)}', ha='center', va='bottom', fontsize=10)

#     # 匯出圖檔
#     plt.savefig(f'{fileName}_data_info.png', dpi=300, bbox_inches='tight')

# count("WineQT.csv")
# count("test.csv")
# count("train.csv")

def plot_combined_files(file_list,output_name="data_info.png"):
    all_counts = {}
    all_qualities = set()

    # 1. 收集所有檔案的數據
    for fileName in file_list:
        df = pd.read_csv(fileName)
        val, cnt = np.unique(df['quality'], return_counts=True)
        cnt = cnt/np.sum(cnt)
        all_counts[fileName] = dict(zip(val, cnt))
        all_qualities.update(val)

    # 排序所有的品質分數，作為 X 軸
    sorted_qualities = sorted(list(all_qualities))
    
    # 2. 設定繪圖參數
    x = np.arange(len(sorted_qualities))  # X 軸基礎位置
    width = 0.25  # 每個長條的寬度
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 7))

    # 3. 繪製每個檔案的長條
    for fileName, counts in all_counts.items():
        # 確保每個分數都有對應數值，若無則補 0
        measurement = [counts.get(q, 0) for q in sorted_qualities]
        
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=fileName)
        ax.bar_label(rects, padding=3,fmt='%.3f')
        multiplier += 1

    # 4. 圖表修飾
    ax.set_xlabel('Quality Score')
    ax.set_ylabel('Count')
    ax.set_title('Quality Distribution Comparison')
    ax.set_xticks(x + width, sorted_qualities) # 讓 X 軸刻度居中
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # 5. 匯出與顯示
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)

plot_combined_files(["WineQT.csv", "train.csv", "test.csv"])