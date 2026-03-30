import pandas as pd

rData = pd.read_csv('WineQT.csv') # row data
# shuffl
rData_shuffled = rData.sample(frac=1, random_state=330).reset_index(drop=True)
# split data
split_point = int(len(rData_shuffled) * 0.8)
train_data = rData_shuffled.iloc[:split_point]
test_data = rData_shuffled.iloc[split_point:]
# generate csv
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False) 