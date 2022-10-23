import numpy as np
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """
    Abstract class which set the functions value(z) and derivative(z) which need to be implemented
    for an activation function
    """

    @abstractmethod
    def value(self, z):
        """
        Value of the activation function in a certain point
        :param z: point to evaluate the function
        :return: value of the function
        """
        pass

    @abstractmethod
    def derivative(self, z):
        """
        Value of the derivative of the activation function
        :param z: point to evaluate the derivative
        :return: value of the derivative
        """
        pass


class Sigmoid(ActivationFunction):
    """
    Sigmoid (or logistic) function sigmoid(x) = 1/(1+exp(-z))
    """
    def value(self, z):
        return (1 / (1 + np.exp(-z)))

    def derivative(self, z):
        return self.value(z) * (1 - self.value(z))


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent, tanh(x)=(exp(x)-exp(-x))/(exp(x)+exp(-x))
    """

    def value(self, z):
        return  ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)))/2

    def derivative(self, z):
        return (1 - self.value(z)**2)/2


class ReLu(ActivationFunction):
    """
    ReLu function relu(x)=max(0,x)
    """

    def value(self, z):
        return np.maximum(0, z)

    def derivative(self, z):
        return (z > 0).astype('int')


class Softmax(ActivationFunction):
    """
    Softmax function, softmax(x)=exp(x)/sum(exp(x))
    """

    def value(self, z):
        return np.exp(z) / sum(np.exp(z))

    def derivative(self, z):
        n = np.max(z.shape)
        J = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    J[i, j] = z[i] * (1 - z[i])
                else:
                    J[i, j] = -z[i] * z[j]
        return J
