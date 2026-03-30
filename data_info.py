import pandas as pd
import numpy as np

def count(fileName):
    print(f'{fileName}:')
    rData = pd.read_csv(fileName) # row data
    rY = rData['quality'].to_numpy() # quality
    val,cnt = np.unique(rY, return_counts=True)
    for i,j in zip(val,cnt):
        print(f'{i} : {j}')

count("WineQT.csv")
count("test.csv")
count("train.csv")