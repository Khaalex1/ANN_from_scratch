import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    """
    Abstract class of a loss function which could be used in a ANN
    """

    @abstractmethod
    def value(self, a, y):
        """
        Value of the loss function
        :param: a : predicted value
        :param: y : ground-truth value
        :return: value of the loss
        """
        pass

    @abstractmethod
    def derivative(self, y, a):
        """
        Value of the derivative of the loss function
        :param: a : predicted value
        :param: y : ground-truth value
        :return: value of the deriavtive of the loss
        """
        pass


class CrossEntropy(Loss):
    """
    Cross entropy function for binary or multi-class classification
    """
    def __init__(self, one_hot=False, binary=False):
        """
        Constructor of the class CrossEntropy
        :param one_hot: Use of one_hot encoding (boolean, default value to False)
        :param binary: Use of binary cross-entropy for binary classification (boolean, default to false)
        """
        self.one_hot = one_hot
        self.binary = binary

    def value(self, y, a):
        if self.one_hot:
            m = y.shape[1]
        else:
            m = y.shape[0]
        if self.binary:
            return(1/m)*np.sum(-y*np.log(a+1e-7) - (1-y)*np.log(1-a+1e-7))

        return (1/m)*np.sum(-y*np.log(a+1e-7))

    def derivative(self, y, a):
        return (a - y) / (a * (1 - a))

    
class MSE(Loss):
    """
    Mean-Squared Error
    """
    def __init__(self, one_hot=False):
        self.one_hot = one_hot

    def value(self, y, a):
        if self.one_hot:
            m = y.shape[1]
        else:
            m = y.shape[0]
        return 1 / (2 * m) * np.sum((y - a) ** 2)

    def derivative(self, y, a):
        m = y.shape[0]
        return 1 / m * np.sum(a - y)

class Abs(Loss):

    def __init__(self, one_hot=False):
        self.one_hot = one_hot

    def value(self, y, a):
        if self.one_hot:
            m = y.shape[1]
        else:
            m = y.shape[0]
        return 1 /m * np.sum(abs(y - a))

    def derivative(self, y, a):
        return np.sign(a - y)