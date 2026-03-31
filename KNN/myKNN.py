import numpy as np
import myUtil

class Basic_KNN():
    def __init__(self,train_file_name:str,distance_fnc):
        self.X_train,self.Y_train=myUtil.read_data(train_file_name)
        self.distance_fnc = distance_fnc
        self.preprocess()
    def judge(test_data,K):
        pass
    def preprocess():
        pass

def judge(test_data,X_train,Y_train,K):
    distance_sq = np.sum((test_data-X_train)**2,axis=1) # calculate distance
    sorted_indices = np.argsort(distance_sq) # sort
    nearest_K = Y_train[sorted_indices[:K]] # select nearest K
    score,count = np.unique(nearest_K, return_counts=True) # vote
    return score[np.argmax(count)]

# preprocess
X_train,Y_train = myUtil.read_data("train.csv")
X_train,(train_mean,train_std) = myUtil.standardization(X_train)

# # get training accuracy
success = 0
for i in range(len(X_train)):
    if judge(X_train[i],X_train,Y_train,10)==Y_train[i]:
        success+=1

print(f"training accuracy: {success/len(X_train)*100:.3f}%")

# get testing accuracy