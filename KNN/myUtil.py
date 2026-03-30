import pandas as pd
import numpy as np

def read_data(fileName):
    rData = pd.read_csv(fileName) # row data
    rX = rData.drop(columns=['quality', 'Id']).values # feature data
    rY = rData['quality'].to_numpy() # quality
    return rX,rY

def preprocess1(rX,rY):
    # normalize
    X = (rX - rX.mean(axis=0))/rX.std(axis=0)
    return X,rY