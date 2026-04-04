import pandas as pd
import numpy as np
import time

def read_data(fileName):
    rData = pd.read_csv(fileName) # row data
    rX = rData.drop(columns=['quality', 'Id']).values # feature data
    rY = rData['quality'].to_numpy() # quality
    return rX,rY

def timeTest(start=[0]):
    res = time.time()-start[0]
    start[0] = time.time()
    return res

def euclidean_distance_sq(test,train,axis=1):
    return np.sum((test-train)**2,axis=axis)

def euclidean_distance(test,train,axis=1):
    return np.sqrt(np.sum((test-train)**2,axis=axis))

def manhattan_distance(test,train,axis=1):
    return np.sum(np.abs(test-train),axis=axis)

def calculateIQR(rX):
    # calculate IQR and return mask
    Q1 = np.percentile(rX,25,axis=0)
    Q3 = np.percentile(rX,75,axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.all((rX>lower_bound) & (rX<upper_bound),axis=1),lower_bound,upper_bound