import numpy as np
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as M

class GaussProxy:
    def __init__(self, trainData, label):
        #替换核函数
        kernel = M(length_scale=1.0,length_scale_bounds=(1e-05,100000.0),nu=2.5)
        self.model = GaussianProcessRegressor(kernel=kernel)
        self.trainData = np.array(trainData)
        self.trianLabel = np.array(label)

    def fit(self):
        return self.model.fit(self.trainData, self.trianLabel)

    def predict(self, testData):

        return self.model.predict(testData)





