import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import matplotlib.pyplot as plt
import ActivationFunction, Loss


def accuracy(predicted, ground_truth):
    """
    Compute the rate of true predictions (accuracy)
    :param predicted: the predicited values
    :param ground_truth: the ground-truth values
    :return: the accuracy (between 0 and 1)
    """
    #TODO : Add the check of conditions to know if it is needed to encode with one_hot or no
    # Maybe it could be more pertinent to add this in the prediciotn function ?
    acc = np.sum(predicted == ground_truth) / ground_truth.shape[0]
    return acc


def one_hot(y, nb_class=2):
    """
    One hot encoding
    :param y: vector to encode
    :param nb_class: number of classes
    :return:
    """
    nb_samp = y.shape[0]
    y_one_hot = np.zeros((nb_class, nb_samp))
    for i in range(nb_samp):
        y_one_hot[int(y[i]), i] = 1
    return y_one_hot


def batchNorm(Z):
    """
    Normalization by centering with the mean and scaling by the standard deviation
    :param Z: the vector to normalize
    :return: the normalized vector
    """
    m, std = np.mean(Z), np.std(Z)
    return (Z - m) / std


def dot_col(y, A):
    C = y.shape[1]
    d = np.zeros((C))
    for col in range(C):
        d[col] = y[:, col].dot(A[:, col])
    return d


class MLP:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.nb_samp = 0
        self.nb_feat = 0
        self.nb_class = 0
        self.Weights = {}
        self.Bias = {}
        self.func_activation = {}
        self.deriv_func = {}
        self.postProcess = {}
        self.error = None
        self.counter = 0

        # Dictionnary with the metrics computed at each epoch
        self.history = {}
        # For the train data the values are computed at each batch and averaged at each epoch
        self.history['acc'] = []
        self.history['loss'] = []
        # For the validaiton data the values are computed at each epoch
        self.history['val_acc'] = []
        self.history['val_loss'] = []
        self.epochs = None

    def fit(self, X, y, loss, BATCH_SIZE=1, EPOCHS=300, val_split = 0.3, l=0.001, reg_L2=0, corr=0.9999, acc_limit=0.99):
        """
        Training of the MLP (fitting the weights with gradient descent)
        :param X: training data matrix(n_samples,n_features)
        :param y: training labels
        :param loss: loss function to use for training (string in ["mse", "abs", "cross_entropy", "binary_cross_entropy"])
        :param BATCH_SIZE: number of samples per batch (default to 1 - Stochastic Gradient Descent)
        :param EPOCHS: number of EPOCHS for training (default to 300)
        :param val_split : percentage of the input data to split into validation data (default to 0.3)
        :param l: learning rate of the gradient descent (default to 0.001)
        :param reg_L2: L2 regularization of the gradient descent (default to 0 - no regularization)
        :param corr: ??
        :return:
        """
        # Check the dimension of the label vector
        if len(y.shape) < 2:
            y = y[:, None]

        #Split the dataset into training and validation dataset
        data_shuffle = np.hstack((X, y))
        np.random.shuffle(data)
        n_val = int(val_split * X.shape[0])
        X_val, y_val = data_shuffle[0:n_val, 0:-1], data_shuffle[0:n_val, -1]
        X_train, y_train = data_shuffle[n_val:-1, 0:-1], data_shuffle[n_val:-1, -1][:, None]
        #Check the inputs
        if self.X_train is not None:  # Why do we need to reinitialize the weights ? Can't we continue the training after one training ?
            self.reinitialize()
        # verify samples (0) > features (1)
        if X.shape[0] < X.shape[1]:
            self.X_train = X_train.T
        else:
            self.X_train = X_train
        self.y_train = y_train

        self.nb_samp = y_train.size
        self.nb_feat = X_train.shape[1]
        self.nb_class = int(y_train.max() + 1)


        self.accuracy = []
        self.loss = []
        # Check the BATCH_SIZE
        if BATCH_SIZE == 0 or BATCH_SIZE > self.y_train.shape[0]:
            BATCH_SIZE = self.y_train.shape[0]
        # Check the loss
        if loss not in ["mse", "abs", "cross_entropy", "binary_cross_entropy"]:
            print("Warning ! Specified loss function is not recognized.")
            print("List of recognized functions : 'cross_entropy', 'abs', mse'.")
            print("Loss function 'cross_entropy' is taken by default.")
            loss = "cross_entropy"
        # Set the loss
        if loss == "mse":
            self.error = Loss.MSE()
        elif loss == "abs":
            self.error = Loss.Abs()
        elif loss == "binary_cross_entropy":
            self.error = Loss.CrossEntropy(one_hot=False, binary=True)
        else:
            self.error = Loss.CrossEntropy(one_hot=False, binary=False)
        step_acc = 0
        step_err = 0
        ep = 0
        print("batch size = ", BATCH_SIZE)
        print("loss function : ", loss)
        print('---'*10)
        print('Training...')
        t0 = datetime.datetime.now()
        while ep < EPOCHS and step_acc<acc_limit:
            #Epoch ep
            ep +=1
            #Gradient descent
            step_acc, step_err = self.gradient_descent(ep, gamma=l, reg_L2=reg_L2, corr=corr, batch_size=BATCH_SIZE)
            #Evaluate the MLP on the validation data
            #TODO : create anonther function or put this part in the accuracy function
            y_pred_val = self.predict(X_val)
            if isinstance(self.func_activation[self.counter], ActivationFunction.Sigmoid):
                val_acc = accuracy((y_pred_val >= 0.5).astype('int'), y_val)
                val_err = self.error.value(y_val, y_pred_val)
            elif isinstance(self.func_activation[self.counter], ActivationFunction.Tanh):
                val_acc = accuracy((y_pred_val >= 0).astype('int'), y_val)
                val_err = self.error.value(y_val, y_pred_val)
            else:
                # softmax
                self.error.one_hot = True
                val_acc = accuracy(np.argmax(y_pred_val, 0), y_val)
                val_err = self.error.value(one_hot(y_val), y_pred_val)
            self.history['val_acc'].append(val_acc)
            self.history['val_loss'].append(val_err)
            #Print the training metrics
            if ep == 1 or ep % 10 == 0:
                print("Epoch : {},  acc : {} - loss : {} - val accuracy : {} - val loss : {}".format(ep, step_acc, step_err, val_acc, val_err))
        tf = datetime.datetime.now() - t0
        print('Training time (hh:mm:ss): {}'.format(tf))
        print('---'*10)
        self.epochs = np.linspace(1, EPOCHS, len(self.history['acc']))
        print('Last epoch :')
        print("Epoch : {},  Batch accuracy : {} - Batch error = {}".format(ep, step_acc, step_err))



    def reinitialize(self):
        """
        Reinitialize the weights of the MLP with random values between -0.5 and +0.5
        """
        for key in self.Weights.keys():
            self.Weights[key] = np.random.rand(self.Weights[key].shape[0], self.Weights[key].shape[1]) - 0.5
            self.Bias[key] = np.random.rand(self.Bias[key].shape[0], self.Bias[key].shape[1]) - 0.5


    def training_curve(self):
        """
        Plot the evolution of the metrics (accuracy and loss) during training
        """
        figure, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title('Accuracy in function of epoch')
        ax[0].plot(self.epochs, self.history['acc'], color='blue', label='train data')
        ax[0].plot(self.epochs, self.history['val_acc'], color='orange', label='val data')
        ax[0].grid()
        ax[0].legend()
        ax[0].set_xlabel('epochs')
        ax[0].set_ylabel('accuracy')
        ax[1].grid()
        ax[1].plot(self.epochs, self.history['loss'], color='blue', label='train data')
        ax[1].plot(self.epochs, self.history['val_loss'], color='orange', label='val data')
        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('error')
        ax[1].set_title('Error (/loss) in function of epoch')
        ax[1].legend()
        figure.tight_layout(pad=3.0)

        plt.show()

    def mini_batches(self, batch_size=1):
        """
        Split the training dataset into 'batch_size' datasets
        :param batch_size: number of samples in each batch (default batch_size=1 - Stochastic Gradient Descent)
        :return: the list of all the batch datasets
        """
        if batch_size > self.nb_samp:
            batch_size = self.nb_samp
            print('Warning : the batch_size parameter is higher than the training dataset size')
            print('     batch_size set by default to the size of the training dataset')
        all_batches = []
        nb_batches = self.nb_samp // batch_size
        # Concatenate the feature matrix with the label vector before shuffle
        data = np.hstack((self.X_train, self.y_train))
        np.random.shuffle(data)  # suffle the data to improve the accuracy
        for i in range(nb_batches):
            if i == nb_batches - 1:
                batch = data[i * batch_size:data.shape[0], :]
            else:
                batch = data[i * batch_size:(i + 1) * batch_size]
            all_batches.append(batch)

        # Return the list of the batch instead of the array because there are not always the same number of samples
        return all_batches

    def add_layer(self, nb_nodes, feat_size=0, activation='tanh'):
        """
        Add a layer to the Multi-Layer Perceptron NN
        :param nb_nodes: number of neurons in the layers
        :param feat_size: input shape ?
        :param activation: activation function of the neurons in the layer
        :return:
        """
        # Update the number of layers
        self.counter += 1
        # Set the activation function of the layer
        if activation not in ['tanh', 'sigmoid', 'relu', 'softmax']:
            activation = 'tanh'
        if activation == 'tanh':
            self.func_activation[self.counter] = ActivationFunction.Tanh()
        elif activation == 'sigmoid':
            self.func_activation[self.counter] = ActivationFunction.Sigmoid()
        elif activation == 'relu':
            self.func_activation[self.counter] = ActivationFunction.ReLu()
        else:
            self.func_activation[self.counter] = ActivationFunction.Softmax()
        #Initialize the weights and bias
        if self.counter == 1:
            self.Weights[1] = np.random.rand(nb_nodes, feat_size) - 0.5
            self.Bias[1] = np.random.rand(nb_nodes, 1) - 0.5
        else:
            last_lay_nodes = self.Weights[self.counter - 1].shape[0]
            self.Weights[self.counter] = np.random.rand(nb_nodes, last_lay_nodes) - 0.5
            self.Bias[self.counter] = np.random.rand(nb_nodes, 1) - 0.5
        print("Added Layer shape : ", self.Weights[self.counter].shape)  # print (input shape output shape) ?

    def BatchNormalization(self):
        self.postProcess[self.counter] = batchNorm


    def forward_propagation(self, X):
        """
        Forward propagation of the samples through the MLP network
        :param X: the feature matrix (n_samples, n_features)
        :return: Z : output of the layers before activation
        :return out : output of the layers after activation
        """
        Z = {}
        out = {}
        out[0] = X
        for key in self.Weights:
            # Forward pass for the layer 'key'
            Z[key] = self.Weights[key] @ out[key - 1] + self.Bias[key]
            out[key] = self.func_activation[key].value(Z[key])
            if key in self.postProcess.keys():
                out[key] = self.postProcess[key](out[key])
        return Z, out


    def back_propagation(self, Z, out, y):
        """
        Backpropagation
        :param Z: the ouputs of the layers before activation
        :param out: the outputs of the layers after activation
        :param y: the labels
        :return:
        """
        #Partial derivatives of the loss
        dZ = {}
        dW = {}
        dB = {}
        for key in reversed(list(Z.keys())):
            if key == self.counter:
                if isinstance(self.func_activation[self.counter], ActivationFunction.Softmax):
                    # Simplification of the partial derivative of the loss in relation of Z with one hot encoding
                    a = out[key]
                    Y = one_hot(y, self.nb_class)
                    dZ[key] = a - Y
                elif isinstance(self.func_activation[self.counter], ActivationFunction.Sigmoid) and isinstance(self.error, Loss.CrossEntropy):
                    if self.error.binary:
                        # Simplification of the partial derivative of the loss in relation of Z
                        a = out[key]
                        dZ[key] = a - y
                    else:
                        # General formula with the chain rule
                        a = out[key]
                        dZ[key] = self.error.derivative(y, a) * self.func_activation[key].derivative(a)
                else:
                    # General formula with the chain rule
                    a = out[key]
                    dZ[key] = self.error.derivative(y, a) * self.func_activation[key].derivative(a)
            else:
                dZ[key] = self.Weights[key + 1].T @ dZ[key + 1] * self.func_activation[key].derivative(Z[key])
            dW[key] = (1 / y.shape[0]) * dZ[key] @ out[key - 1].T
            dB[key] = (1 / y.shape[0]) * np.sum(dZ[key])

        return dW, dB

    def update_parameters(self, dW, dB, gamma, batch_size=1, reg_L2=0):
        """
        Update the parameters of the MLP
        :param dW: partial derivative of the cost in relation of the weights W
        :param dB: partial derivative of the cost in relation of the bias B
        :param gamma: learning rate
        :param batch_size: number of samples for each batch (default to 1 - Stochastic Gradient Descent)
        :param reg_L2:
        :return:
        """
        for key in self.Weights:
            self.Weights[key] += - gamma * dW[key] + (reg_L2 / batch_size) * self.Weights[key]
            self.Bias[key] -= gamma * dB[key]
        return self.Weights, self.Bias


    def gradient_descent(self, ep, gamma=0.1,
                         reg_L2=0, corr=0.9999, batch_size=1):
        """
        General gradient descent algorithm
        :param gamma: learning rate
        :param epochs: number of epochs
        :param acc: maximum accuracy to stop the training
        :param reg_L2: L2 regularization
        :param corr:
        :param loss: Loss function to use
        :param batch_size: number of samples in each batch (default to 1 - Stochastic Gradient Descent)
        :return:
        """

        all_batches = self.mini_batches(batch_size)
        simple_acc = 0
        simple_err = 0
        for i in range(len(all_batches)):
            # Get the train set for the batch i
            data_batch = all_batches[i]
            X_batch = data_batch[:, :-1]
            Y_batch = data_batch[:, -1]

            # Forward pass
            Z, out = self.forward_propagation(X_batch.T)
            # Backpropagation
            dW, dB = self.back_propagation(Z, out, Y_batch)
            self.update_parameters(dW, dB, gamma * corr ** ep, reg_L2=reg_L2, batch_size=batch_size)
            y = Y_batch
            a = out[self.counter]
            if isinstance(self.func_activation[self.counter], ActivationFunction.Sigmoid):
                simple_acc += accuracy((a >= 0.5).astype('int'), y)
                simple_err += self.error.value(y, a)
            elif isinstance(self.func_activation[self.counter], ActivationFunction.Tanh):
                simple_acc += accuracy((a >= 0).astype('int'), y)
                simple_err += self.error.value(y, a)
            else:
                # softmax
                self.error.one_hot = True
                simple_acc += accuracy(np.argmax(a, 0), y)
                simple_err += self.error.value(one_hot(y), a)
        step_acc = simple_acc / (i + 1)  # i = len(all_batches) ?
        step_err = simple_err / (i + 1)
        self.history['acc'].append(step_acc)
        self.history['loss'].append(step_err)
        return step_acc, step_err


    def predict(self, X):
        """
        Prediction of the MLP Neural Network
        :param X: input of the MLP
        :return: the predictions (posterior probabilities in classification)
        """
        if len(X.shape) < 2:
            X = X[:, None]
        else:
            if X.shape[0] > X.shape[1]:
                X = X.T
        Z, out = self.forward_propagation(X)
        return out[self.counter]


#  ------------------------------------------------------
# TEST
# --------------------------------------------------------
data = pd.read_csv('../wdbc.csv')
data.columns = [i for i in range(32)]

d = {'B': 0, 'M': 1}
label_col = data[1].replace(d)

data.pop(1)
data['label'] = label_col

print(data.shape)
print(data.head())

print(np.where(data.isna() == True))

data = data.sample(frac=1)
data = data.to_numpy()

nb_samp, nb_feat = data.shape
nb_feat -= 1
nb_train = int(0.8 * nb_samp)

X_train = data[:nb_train, :nb_feat]
# instance normalization
X_train /= X_train.max(axis=0)

Y_train = data[:nb_train, nb_feat].astype('int')

X_test = data[nb_train:, :nb_feat]
X_test /= X_test.max(axis=0)
Y_test = data[nb_train:, nb_feat].astype('int')

AN = MLP()
AN.add_layer(nb_nodes=64, feat_size=nb_feat, activation='relu')
AN.BatchNormalization()
AN.add_layer(nb_nodes=32, activation='relu')
#AN.BatchNormalization()
# AN.add_layer(nb_nodes=100, activation='relu')
# AN.BatchNormalization()
# AN.add_layer(nb_nodes=100, activation='relu')
# AN.BatchNormalization()
# AN.add_layer(nb_nodes=100, activation='relu')
# AN.BatchNormalization()
# AN.add_layer(nb_nodes=50, activation='relu')
AN.BatchNormalization()
AN.add_layer(nb_nodes=2, activation='softmax')
AN.fit(X_train, Y_train, loss="cross_entropy", BATCH_SIZE=32, EPOCHS=350, l=0.001)
#AN.gradient_descent(gamma=1e-3, epochs=400, loss="cross_entropy", batch_size=16)
AN.training_curve()

y_pred = np.argmax(AN.predict(X_test), 0)
print('---'*10)
print("Accuracy on Test set :", accuracy(y_pred, Y_test))
# use the same conditions to compute the accuracy for the test as for the training and val data (check the instance and encode in one hot if needed)