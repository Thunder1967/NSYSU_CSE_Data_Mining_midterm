import numpy as np
import myUtil

class STD_Preprocess():
    def __init__(self):
        self.mean = None
        self.std = None
    def trainPreprocess(self,rX,rY):
        # Standardization
        self.mean = rX.mean(axis=0)
        self.std = rX.std(axis=0)
        X = (rX - self.mean)/self.std
        return X,rY
    def testPreprocess(self,rX):
        return (rX - self.mean)/self.std
      
class STD_IQR_Preprocess(STD_Preprocess):
    def __init__(self,lower_bound=25,upper_bound=75):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    def trainPreprocess(self,rX,rY):
        # remove outliers
        mask = myUtil.calculateIQR(rX,self.lower_bound,self.upper_bound)
        print(f'remove {len(mask)-np.sum(mask)} outliers')
        return super().trainPreprocess(rX[mask],rY[mask])

class Scale_Preprocess():
    def __init__(self):
        self.min = None
        self.max = None
    def trainPreprocess(self,rX,rY):
        # Min-Max Scaling
        self.min = np.min(rX, axis=0)
        self.max = np.max(rX, axis=0)
        X = (rX - self.min)/(self.max - self.min)
        return X,rY
    def testPreprocess(self,rX):
        return (rX - self.min)/(self.max - self.min)

class Scale_IQR_Preprocess(Scale_Preprocess):
    def __init__(self,lower_bound=25,upper_bound=75):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    def trainPreprocess(self,rX,rY):
        mask = myUtil.calculateIQR(rX,self.lower_bound,self.upper_bound)
        print(f'remove {len(mask)-np.sum(mask)} outliers')
        return super().trainPreprocess(rX[mask],rY[mask])

if __name__ == "__main__":
    import myUtil
    a = STD_IQR_Preprocess()
    X,Y=myUtil.read_data("train.csv")
    print(len(a.remove_outliers(X)))