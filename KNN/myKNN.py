import numpy as np
import myUtil
import myPreprocess

class BasicKNN():
    def __init__(self,train_file_name:str,preprocess,distance_fnc,K):
        # init
        self.X_train,self.Y_train=myUtil.read_data(train_file_name)
        self.preprocess = preprocess
        self.distance_fnc = distance_fnc
        self.K = K
        # preprocess
        self.X_train =  self.preprocess.trainPreprocess(self.X_train)

    def setK(self,K):
        self.K = K

    def judge(self,test_data):
        pass

    def getAccuracy(self,X_test,Y_test):
        success = 0
        total = len(X_test)
        for i in range(total):
            if self.judge(X_test[i])==Y_test[i]:
                success+=1
        return success/total
    
    def getTrainingAccuracy(self):
        return self.getAccuracy(self.X_train,self.Y_train)
    
    def getTestingAccuracy(self,X_test,Y_test):
        X_test = self.preprocess.testPreprocess(X_test)
        return self.getAccuracy(X_test,Y_test)

class BruteKNN(BasicKNN):
    def __init__(self,train_file_name:str,preprocess,distance_fnc,defaultK=10):
        super().__init__(train_file_name,preprocess,distance_fnc,defaultK)
    def judge(self,test_data):
        distance_sq = self.distance_fnc(test_data,self.X_train) # calculate distance
        sorted_indices = np.argsort(distance_sq) # sort
        nearest_K = self.Y_train[sorted_indices[:self.K]] # select nearest K
        score,count = np.unique(nearest_K, return_counts=True) # vote
        return score[np.argmax(count)]

if __name__ == '__main__':
    KNN1 = BruteKNN("train.csv",myPreprocess.STDPreprocess(),myUtil.euclidean_distance_sq,10)
    print(f"training accuracy: {KNN1.getTrainingAccuracy()*100:.3f}%")
    print(f"testing accuracy: {KNN1.getTestingAccuracy(*myUtil.read_data("test.csv"))*100:.3f}%")