import numpy as np

class STDPreprocess():
    def __init__(self):
        self.mean = None
        self.std = None
    def trainPreprocess(self,rX):
        # Standardization
        self.mean = rX.mean(axis=0)
        self.std = rX.std(axis=0)
        X = (rX - self.mean)/self.std
        return X
    def testPreprocess(self,rX):
        return (rX - self.mean)/self.std