import pandas as pd
import numpy as np

def read_data(fileName):
    rData = pd.read_csv(fileName) # row data
    rX = rData.drop(columns=['quality', 'Id']).values # feature data
    rY = rData['quality'].to_numpy() # quality
    return rX,rY

def euclidean_distance_sq(test,train):
    return np.sum((test-train)**2,axis=1)

def manhattan_distance(test,train):
    return np.sum(np.abs(test-train),axis=1)

def calculateIQR(rX):
    # calculate IQR and return mask
    Q1 = np.percentile(rX,25,axis=0)
    Q3 = np.percentile(rX,75,axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.all((rX>lower_bound) & (rX<upper_bound),axis=1),lower_bound,upper_bound