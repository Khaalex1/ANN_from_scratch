import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):

    @abstractmethod
    def batch_size(self, size=32):
        pass

    @abstractmethod
    def update(self, key, dW, dB):
        pass

    @abstractmethod
    def weight_factor(self, key, dW, epoch):
        pass

    @abstractmethod
    def bias_factor(self, key, dB, epoch):
        pass

    @abstractmethod
    def initialize(self):
        pass

class RMSPROP(Optimizer):
    """
    RMSPROP optimizer
    """

    def __init__(self, MLP, beta_2 = 0.9, eps = 1e-8):
        self.MLP = MLP
        self.Sb = {}
        self.Sw = {}
        self.beta_2 = beta_2
        self.eps = eps

    def initialize(self):
        self.Sb = {key: 0 for key in self.MLP.Weights.keys()}
        self.Sw = {key: 0 for key in self.MLP.Weights.keys()}

    def batch_size(self, size=32):
        if size <= 0 or size > self.MLP.X_train.shape[0]:
            self.MLP.batch_size = self.MLP.X_train.shape[0]
        else:
            self.MLP.batch_size = size


    def update(self, key, dW, dB):
        self.Sw[key] = self.beta_2 * self.Sw[key] + (1 - self.beta_2) * dW[key] ** 2
        self.Sb[key] = self.beta_2 * self.Sb[key] + (1 - self.beta_2) * dB[key] ** 2

    def weight_factor(self, key, dW, t):
        return dW[key]  * 1 / np.sqrt(self.Sw[key] + self.eps)

    def bias_factor(self, key, dB, t):
        return dB[key] * 1/np.sqrt(self.Sb[key] + self.eps)



class ADAM(Optimizer):
    """
    ADAM optimizer
    """

    def __init__(self , MLP, beta_1 = 0.9, beta_2 = 0.9, eps = 1e-8):
        self.MLP = MLP
        self.Sb = {}
        self.Sw = {}
        self.Vb = {}
        self.Vw = {}
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    def initialize(self):
        self.Sb = {key: 0 for key in self.MLP.Weights.keys()}
        self.Sw = {key: 0 for key in self.MLP.Weights.keys()}
        self.Vb = {key: 0 for key in self.MLP.Weights.keys()}
        self.Vw = {key: 0 for key in self.MLP.Weights.keys()}

    def batch_size(self, size=32):
        if size <= 0 or size > self.MLP.X_train.shape[0]:
            self.MLP.batch_size = self.MLP.X_train.shape[0]
        else:
            self.MLP.batch_size = size

    def update(self, key, dW, dB):
        self.Vw[key] = self.beta_1 * self.Vw[key] + (1 - self.beta_1) * dW[key]
        self.Vb[key] = self.beta_1 * self.Vb[key] + (1 - self.beta_1) * dB[key]
        self.Sw[key] = self.beta_2 * self.Sw[key] + (1 - self.beta_2) * dW[key] ** 2
        self.Sb[key] = self.beta_2 * self.Sb[key] + (1 - self.beta_2) * dB[key] ** 2

    def weight_factor(self, key, dW, t):
        Vw_corr = self.Vw[key] / (1 - self.beta_1 ** t)
        Sw_corr = self.Sw[key] / (1 - self.beta_2 ** t)
        return Vw_corr * 1 / np.sqrt(Sw_corr + self.eps)


    def bias_factor(self, key, dB, t):
        Vb_corr = self.Vb[key] / (1 - self.beta_1 ** t)
        Sb_corr = self.Sb[key] / (1 - self.beta_2 ** t)
        return Vb_corr * 1 / np.sqrt(Sb_corr + self.eps)



class Minibatch(Optimizer):
    """
    Classic gradient descent
    """

    def __init__(self, MLP):
        self.MLP = MLP

    def batch_size(self, size=32):
        if size <= 0 or size > self.MLP.X_train.shape[0]:
            self.MLP.batch_size = self.MLP.X_train.shape[0]
        else:
            self.MLP.batch_size = size

    def update(self, key, dW, dB):
        pass

    def weight_factor(self, key, dW, t):
        return dW[key]

    def bias_factor(self, key, dB, t):
        return dB[key]

    def initialize(self):
        pass

class SGD(Minibatch):
    """
    SGD
    """
    def __init__(self , MLP):
        super().__init__(MLP)

    def batch_size(self, size=32):
        self.MLP.batch_size = 1


class Batch(Minibatch):
    """
    Classic gradient descent
    """

    def __init__(self, MLP):
        super().__init__(MLP)

    def batch_size(self, size=32):
        self.MLP.batch_size = self.MLP.X_train.shape[0]

