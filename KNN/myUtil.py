import pandas as pd
import numpy as np

def read_data(fileName):
    rData = pd.read_csv(fileName) # row data
    rX = rData.drop(columns=['quality', 'Id']).values # feature data
    rY = rData['quality'].to_numpy() # quality
    return rX,rY

def euclidean_distance_sq(test,train):
    return np.sum((test-train)**2,axis=1)
