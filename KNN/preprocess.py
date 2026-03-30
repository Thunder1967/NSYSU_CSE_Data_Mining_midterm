import numpy as np
import pandas as pd

def read_data():
    rData = pd.read_csv('WineQT.csv') # row data
    rX = rData.drop(columns=['quality', 'Id']).values # feature data
    rY = rData['quality'].to_numpy() # quality
    return rX,rY

def preprocess1():
    rX,rY = read_data()

    # normalize
    X = (rX - rX.mean(axis=0))/rX.std(axis=0)
    return X,rY