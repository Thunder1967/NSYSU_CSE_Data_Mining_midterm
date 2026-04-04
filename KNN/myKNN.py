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
        self.X_train,self.Y_train =  self.preprocess.trainPreprocess(self.X_train,self.Y_train)

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
        distance = self.distance_fnc(test_data,self.X_train) # calculate distance
        sorted_indices = np.argsort(distance) # sort
        nearest_K = self.Y_train[sorted_indices[:self.K]] # select nearest K
        score,count = np.unique(nearest_K, return_counts=True) # vote
        return score[np.argmax(count)]

class BallTreeKNN(BasicKNN):
    class Node:
        def __init__(self, X, indices, distance_fnc, leaf_size=20):
            self.indices = indices
            self.distance_fnc = distance_fnc
            cur_X = X[indices]
            self.center = np.mean(cur_X,axis=0)
            dist_to_center = self.distance_fnc(self.center,cur_X) # (1,)
            pointA_arg = np.argmax(dist_to_center) # Farthest point
            self.radius = dist_to_center[pointA_arg]
            self.is_leaf = len(indices)<=leaf_size

            if not self.is_leaf:
                # recursive create node
                pointA = cur_X[pointA_arg]
                dist_to_pA = self.distance_fnc(pointA,cur_X)
                pointB = cur_X[np.argmax(dist_to_pA)]
                # projection and build balance tree
                vector_AB = pointB-pointA
                projection = np.dot(cur_X,vector_AB)
                median = np.median(projection)
                mask = projection < median

                self.left = BallTreeKNN.Node(X, indices[mask], self.distance_fnc, leaf_size)
                self.right = BallTreeKNN.Node(X, indices[~mask], self.distance_fnc, leaf_size)
            else:
                self.left = None
                self.right = None

    def __init__(self,train_file_name:str,preprocess,distance_fnc,defaultK=10,leaf_size=20):
        super().__init__(train_file_name,preprocess,distance_fnc,defaultK)
        self.root = BallTreeKNN.Node(self.X_train, np.arange(len(self.X_train)), self.distance_fnc, leaf_size)
    def judge(self,test_data):
        best_neighbor_indices = np.array([], dtype=int)
        best_neighbors = np.array([], dtype=float)
        def search(node:BallTreeKNN.Node):
            nonlocal best_neighbor_indices,best_neighbors
            if len(best_neighbors)==self.K and \
            self.distance_fnc(test_data,node.center,axis=0)-node.radius>=best_neighbors[-1]:
                return
            if node.is_leaf:
                # test data in ball
                new_indices = np.concatenate([best_neighbor_indices,node.indices])
                new_dictance = np.concatenate([
                    best_neighbors,
                    self.distance_fnc(test_data,self.X_train[node.indices])])
                
                sorted_new_dictance_arg = np.argsort(new_dictance)[:self.K]

                best_neighbor_indices = new_indices[sorted_new_dictance_arg]
                best_neighbors = new_dictance[sorted_new_dictance_arg]
            else:
                # go down and search possible node
                left = node.left
                right = node.right
                if self.distance_fnc(test_data,left.center,axis=0)<\
                self.distance_fnc(test_data,right.center,axis=0):
                    search(left)
                    search(right)
                else:
                    search(right)
                    search(left)
        
        search(self.root)
        nearest_K = self.Y_train[best_neighbor_indices] # select nearest K
        score,count = np.unique(nearest_K, return_counts=True) # vote
        return score[np.argmax(count)]
    
if __name__ == '__main__':
    # for testing
    myUtil.timeTest()
    KNN1 = BruteKNN("train.csv",myPreprocess.STD_Preprocess(),myUtil.euclidean_distance_sq,10)
    print(f"training accuracy: {KNN1.getTrainingAccuracy()*100:.3f}%")
    print(f"testing accuracy: {KNN1.getTestingAccuracy(*myUtil.read_data("test.csv"))*100:.3f}%")
    print(f'It costs {myUtil.timeTest()} second')

    myUtil.timeTest()
    KNN1 = BallTreeKNN("train.csv",myPreprocess.STD_Preprocess(),myUtil.euclidean_distance,10)
    print(f"training accuracy: {KNN1.getTrainingAccuracy()*100:.3f}%")
    print(f"testing accuracy: {KNN1.getTestingAccuracy(*myUtil.read_data("test.csv"))*100:.3f}%")
    print(f'It costs {myUtil.timeTest()} second')