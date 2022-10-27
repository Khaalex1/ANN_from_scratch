import numpy as np

class MinMax:
    """
    MinMax rescaling class
    """
    def __init__(self, new_max = 1, new_min =0):
        self.old_min = None
        self.old_max = None
        self.new_min = new_min
        self.new_max = new_max

    def fit(self, X):
        """
        Initializing the the 'old' min max values
        :param X: Data (2D array) to fit
        """
        self.old_min = X.min(axis=0)
        self.old_max = X.max(axis=0)

    def transform(self, X):
        """
        Transforming the data
        :param X: Data (2D array) to transform
        :return: Transformed array
        """
        return ((X - self.old_min) / (self.old_max - self.old_min))*(self.new_max - self.new_min) + self.new_min

    def fit_transform(self, X):
        """
        Fit the data and rescling it
        :param X: Data (2D array) to transform
        :return: Transformed array
        """
        self.old_min = X.min(axis=0)
        self.old_max = X.max(axis=0)
        return self.transform(X)