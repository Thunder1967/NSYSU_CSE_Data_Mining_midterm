import numpy as np
import pandas as pd

rData = pd.read_csv('WineQT.csv') # row data
fData = rData.drop(columns=['quality', 'Id']).values # feature data