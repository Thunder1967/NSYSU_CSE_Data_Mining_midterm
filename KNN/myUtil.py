import pandas as pd

def read_data():
    rData = pd.read_csv('WineQT.csv') # row data
    rX = rData.drop(columns=['quality', 'Id']).values # feature data
    rY = rData['quality'].to_numpy() # quality
    return rX,rY