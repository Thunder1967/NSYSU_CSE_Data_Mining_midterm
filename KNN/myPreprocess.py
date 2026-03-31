import numpy as np

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
    def __init__(self,lower=25,upper=75):
        super().__init__()
        self.lower = lower
        self.upper = upper
    def trainPreprocess(self,rX,rY):
        # calculate IQR
        Q1 = np.percentile(rX,self.lower,axis=0)
        Q3 = np.percentile(rX,self.upper,axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # remove outliers
        mask = np.all((rX>lower_bound) & (rX<upper_bound),axis=1)
        print(f'remove {len(mask)-np.sum(mask)} outliers')
        return super().trainPreprocess(rX[mask],rY[mask])


if __name__ == "__main__":
    import myUtil
    a = STDPreprocess2()
    X,Y=myUtil.read_data("train.csv")
    print(len(a.remove_outliers(X)))