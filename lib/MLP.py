import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import time
import matplotlib.pyplot as plt
import ActivationFunction, Loss
from MinMax import MinMax
import Optimizer


def accuracy(predicted, ground_truth):
    """
    Compute the rate of true predictions (accuracy)
    :param predicted: the predicited values
    :param ground_truth: the ground-truth values
    :return: the accuracy (between 0 and 1)
    """
    acc = np.sum(predicted == ground_truth) / ground_truth.shape[0]
    return acc


def one_hot(y, nb_class=2):
    """
    One hot encoding
    :param y: label vector to encode
    :param nb_class: number of classes
    :return: one hot encoded vector of shape (nb_class, nb_samples)
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
    if std ==0:
        return Z
    else:
        return (Z - m) / std

def K_fold_separation(X, y, K=10):
    """
    K fold separation of training set
    :param X: Training set
    :param y: Label vector
    :param K: Number of folds acquired from the K fold separation
    :return: Train_sets, Train_y 3D arrays
             Test_sets, Test_y 2D arrays
             The 1st dim corresponds to each fold
    """
    nb_samp = y.size
    test_size = nb_samp // K
    Test_sets = []
    Test_y = []
    Train_sets = []
    Train_y = []
    for i in range(K):
        Test_sets.append(X[i * test_size:(i + 1) * test_size, :])
        Test_y.append(y[i * test_size:(i + 1) * test_size])
        if i == 0:
            Train_sets.append(X[(i + 1) * test_size:, :])
            Train_y.append(y[(i + 1) * test_size:])
        elif i == (K - 1):
            Train_sets.append(X[:(i) * test_size, :])
            Train_y.append(y[:(i) * test_size])
        else:
            X_train = np.vstack((X[:i * test_size, :], X[(i + 1) * test_size:, :]))
            if len(y.shape) < 2:
                Y_train = np.hstack((y[:(i) * test_size], y[(i + 1) * test_size:]))
            else:
                Y_train = np.vstack((y[:(i) * test_size], y[(i + 1) * test_size:]))
            Train_sets.append(X_train)
            Train_y.append(Y_train)
    return Train_sets, Train_y, Test_sets, Test_y


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

        self.loss = None
        self.optimizer = None
        self.opt_class = None
        self.batch_size = 0

    def compile(self, optimizer="minibatch", loss = 'cross_entropy'):
        if optimizer not in ["sgd", "batch", "minibatch", "rmsprop", "adam"]:
            print("Warning ! Specified optimizer is not recognized.")
            print("List of recognized optimizers : 'sgd', 'batch', 'minibatch', 'rmsprop', 'adam'.")
            print("Optimizer 'minibatch' is taken by default.")
            optimizer = 'minibatch'
        self.optimizer = optimizer
        if optimizer == "sgd":
            self.opt_class = Optimizer.SGD(self)
        elif optimizer == "batch":
            self.opt_class = Optimizer.Batch(self)
        elif optimizer == "minibatch":
            self.opt_class = Optimizer.Minibatch(self)
        elif optimizer == "rmsprop":
            self.opt_class = Optimizer.RMSPROP(self)
        else:
            self.opt_class = Optimizer.ADAM(self)

        # Check the loss
        if loss not in ["mse", "abs", "cross_entropy", "binary_cross_entropy"]:
            print("Warning ! Specified loss function is not recognized.")
            print("List of recognized functions : 'cross_entropy', 'abs', mse'.")
            print("Loss function 'cross_entropy' is taken by default.")
            loss = "cross_entropy"
        self.loss = loss
        # Set the loss
        if loss == "mse":
            self.error = Loss.MSE()
        elif loss == "abs":
            self.error = Loss.Abs()
        elif loss == "binary_cross_entropy":
            self.error = Loss.CrossEntropy(one_hot=False, binary=True)
        else:
            self.error = Loss.CrossEntropy(one_hot=False, binary=False)



    def fit(self, X, y, BATCH_SIZE=1, EPOCHS=300, val_split=0.3, l=0.1, print_res = True):
        """
        Training of the MLP (fitting the weights with gradient descent)
        :param X: training data matrix(n_samples,n_features)
        :param y: training labels
        :param loss: loss function to use for training (string in ["mse", "abs", "cross_entropy", "binary_cross_entropy"])
        :param BATCH_SIZE: number of samples per batch (default to 1 - Stochastic Gradient Descent)
        :param EPOCHS: number of EPOCHS for training (default to 300)
        :param val_split : percentage of the input data to split into validation data (default to 0.3)
        :param l: learning rate of the gradient descent (default to 0.001)
        :return:
        """

        #Warning
        if isinstance(self.func_activation[self.counter], ActivationFunction.Tanh):
            print("WARNING ! TANH ACTIVATION ON OUTPUT LAYER")
        if self.Weights[self.counter].shape[0]>1 and not(isinstance(self.func_activation[self.counter], ActivationFunction.Softmax)):
            print("WARNING ! ATTEMPTED TO USE OTHER OUTPUT ACTIVATION THAN SOFTMAX WHILE OUTPUT SIZE > 1")
            print("SOFTMAX TAKEN BY DEFAULT")
            self.func_activation[self.counter] = ActivationFunction.Softmax()

        # Check the dimension of the label vector
        if len(y.shape) < 2:
            y = y[:, None]

        # Split the dataset into training and validation dataset
        if val_split:
            data_shuffle = np.hstack((X, y))
            np.random.shuffle(data_shuffle)
            n_val = int(val_split * X.shape[0])
            X_val, y_val = data_shuffle[0:n_val, 0:-1], data_shuffle[0:n_val, -1]
            X_train, y_train = data_shuffle[n_val:-1, 0:-1], data_shuffle[n_val:-1, -1][:, None]
        else:
            X_train, y_train = X, y

        # Check the input
        # verify samples (0) > features (1)
        if X_train.shape[0] < X_train.shape[1]:
            self.X_train = X_train.T
        else:
            self.X_train = X_train
        self.y_train = y_train

        self.nb_samp = y_train.size
        self.nb_feat = X_train.shape[1]
        self.nb_class = int(y_train.max() + 1)

        # Check the BATCH_SIZE
        self.opt_class.initialize()
        self.opt_class.batch_size(BATCH_SIZE)
        step_acc = 0
        step_err = 0
        ep = 0
        if print_res:
            print("batch size = ", self.batch_size)
            print("loss function : ", self.loss)
            print("optimizer : ", self.optimizer)
            print('---' * 10)
            print('Training...')
        t0 = datetime.datetime.now()
        while ep < EPOCHS :
            # Epoch ep
            ep += 1
            # Gradient descent
            step_acc, step_err = self.gradient_descent(ep, gamma=l, batch_size=self.batch_size)
            # Evaluate the MLP on the validation data
            if val_split:
                y_proba_val = self.predict_proba(X_val)
                y_pred_val = self.predict(X_val)
                val_acc = accuracy(y_pred_val, y_val)
                if isinstance(self.func_activation[self.counter], ActivationFunction.Softmax):
                    self.error.one_hot = True
                    val_err = self.error.value(one_hot(y_val), y_proba_val)
                else :
                    val_err = self.error.value(y_val, y_proba_val)
                self.history['val_acc'].append(val_acc)
                self.history['val_loss'].append(val_err)
            if print_res:
                # Print the training metrics
                if ep == 1 or ep % 10 == 0:
                    if val_split:
                        print("Epoch : {},  acc : {} - loss : {} - val accuracy : {} - val loss : {}".format(ep, step_acc,step_err, val_acc, val_err))
                    else:
                        print("Epoch : {},  acc : {} - loss : {} ".format(ep, step_acc,step_err))
        if print_res:
            tf = datetime.datetime.now() - t0
            print('Training time (hh:mm:ss): {}'.format(tf))
            print('---' * 10)
            print('Last epoch :')
            print("Epoch : {},  Batch accuracy : {} - Batch error = {}".format(ep, step_acc, step_err))
        return self.history


    def reinitialize(self):
        """
        Reinitialize the weights of the MLP with random values between -0.5 and +0.5
        """
        for key in self.Weights.keys():
            self.Weights[key] = np.random.rand(self.Weights[key].shape[0], self.Weights[key].shape[1]) - 0.5
            self.Bias[key] = np.random.rand(self.Bias[key].shape[0], self.Bias[key].shape[1]) - 0.5
        self.opt_class.initialize()


    def training_curve(self):
        """
        Plot the evolution of the metrics (accuracy and loss) during training
        """
        figure, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title('Accuracy in function of epoch \n loss = {},\n optimizer = {},\n batch size = {}'.format(self.loss, self.optimizer, self.batch_size))
        ax[0].plot(self.history['acc'], color='blue', label='train data')
        ax[0].plot(self.history['val_acc'], color='orange', label='val data')
        ax[0].grid()
        ax[0].legend()
        ax[0].set_xlabel('epochs')
        ax[0].set_ylabel('accuracy')
        ax[1].grid()
        ax[1].plot(self.history['loss'], color='blue', label='train data')
        ax[1].plot(self.history['val_loss'], color='orange', label='val data')
        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('error')
        ax[1].set_title('Error in function of epoch \n loss = {},\n optimizer = {},\n batch size = {}'.format(self.loss, self.optimizer, self.batch_size))
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
        np.random.shuffle(data)  # shuffle the data to improve the accuracy
        for i in range(nb_batches):
            if i == nb_batches - 1:
                batch = data[i * batch_size:data.shape[0], :]
            else:
                batch = data[i * batch_size:(i + 1) * batch_size]
            all_batches.append(batch)

        # Return the list of the batch instead of the array because there are not always the same number of samples
        return all_batches

    def add_layer(self, nb_nodes, feat_size=0, activation='sigmoid'):
        """
        Add a layer to the Multi-Layer Perceptron NN
        :param nb_nodes: number of neurons in the layers
        :param feat_size: input shape (number of features)
        :param activation: activation function of the neurons in the layer
        :return:
        """
        # Update the number of layers
        self.counter += 1

        # Impossible case
        if nb_nodes == 1 and activation == "softmax":
            print("Warning ! Softmax is not to be used on 1 output layer. Sigmoid is taken by default")
            activation = "sigmoid"

        # Set the activation function of the layer
        if activation not in ['tanh', 'sigmoid', 'relu', 'softmax']:
            activation = 'sigmoid'
        if activation == 'tanh':
            self.func_activation[self.counter] = ActivationFunction.Tanh()
        elif activation == 'sigmoid':
            self.func_activation[self.counter] = ActivationFunction.Sigmoid()
        elif activation == 'relu':
            self.func_activation[self.counter] = ActivationFunction.ReLu()
        else:
            self.func_activation[self.counter] = ActivationFunction.Softmax()
        # Initialize the weights and bias
        if self.counter == 1:
            self.Weights[1] = np.random.rand(nb_nodes, feat_size) - 0.5
            self.Bias[1] = np.random.rand(nb_nodes, 1) - 0.5
        else:
            last_lay_nodes = self.Weights[self.counter - 1].shape[0]
            self.Weights[self.counter] = np.random.rand(nb_nodes, last_lay_nodes) - 0.5
            self.Bias[self.counter] = np.random.rand(nb_nodes, 1) - 0.5
        print("Added Layer shape : ", self.Weights[self.counter].shape)

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

    def update_parameters(self, dW, dB, gamma, iteration):
        """
        Update the parameters of the MLP
        :param dW: partial derivative of the cost in relation of the weights W
        :param dB: partial derivative of the cost in relation of the bias B
        :param gamma: learning rate
        :return:
        """
        for key in self.Weights:
            self.opt_class.update(key, dW, dB)
            self.Weights[key] -= gamma * self.opt_class.weight_factor(key, dW, iteration)
            self.Bias[key] -= gamma * self.opt_class.bias_factor(key, dB, iteration)
        return self.Weights, self.Bias


    def gradient_descent(self, epoch, gamma=0.001,
                          batch_size=1):
        """
        General gradient descent algorithm
        :param gamma: learning rate
        :param epochs: number of epochs
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
            t = self.batch_size*epoch + i
            self.update_parameters(dW, dB, gamma, t)
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
        step_acc = simple_acc / (len(all_batches) + 1)
        step_err = simple_err / (len(all_batches) + 1)
        self.history['acc'].append(step_acc)
        self.history['loss'].append(step_err)
        return step_acc, step_err


    def predict_proba(self, X):
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

    def predict(self, X):
        """
        Prediction in terms of class of the MLP Neural Network
        :param X: input of the MLP
        :return: predicted labels. Threshold of 0.5 for logistic/pseudo-tanh regression
        Warning ! Tanh output activation is allowed but is not recommended
        as the output will be between -1 and 1.
        Moreover losses like cross entropy cannot compute tanh negetive values
        """
        y_proba = self.predict_proba(X)
        if isinstance(self.func_activation[self.counter], ActivationFunction.Sigmoid):
            return (y_proba >= 0.5).astype('int')
        elif isinstance(self.func_activation[self.counter], ActivationFunction.Tanh):
            return (y_proba >= 0).astype('int')
        else:
            # softmax
            return np.argmax(y_proba, 0)

    def Kfold_simulation(self, X, y, Kfold=10, BATCH_SIZE=32, EPOCHS=300, l=0.1, print_res = False):
        """
        Simulate the K fold cross validation on the data. At each step (0 to K-1),
        accuracy on current test set is displayed.
        :param X: Train set / feature array (2D)
        :param y: Label vector (1D)
        :Kfold: Number of separate folds wanted from original data
        :BATCH_SIZE: batch size
        :EPOCHS: Training number of epochs
        :l: learning rate
        :return: Average accuracy on all test sets
        """
        Train_sets, Train_y, Test_sets, Test_y = K_fold_separation(X, y, K=Kfold)
        acc = []
        times = []
        for i in range(Kfold):
            self.reinitialize()
            t0 = time.time()
            self.fit(Train_sets[i], Train_y[i], BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS, l=l, print_res=print_res, val_split=0.0)
            tf = time.time() - t0
            times.append(tf)
            y_pred = self.predict(Test_sets[i])
            acci = accuracy(y_pred, Test_y[i])
            acc.append(acci)
            print("Test set {}: Accuracy = {} ".format(i, acci))

        print("Average accuracy by X-validation = {} +-  {}".format(np.round(np.mean(acc), 4), np.round(np.std(acc), 4)))
        return acc, times

def hidden_layers_tunning(X, y, nb_hidden_layers, nodes_per_layers, BATCH_SIZE, EPOCH, optimizer='minibatch', l=0.01):
    history_kfold = {}
    times_kfold = {}
    for i in range(len(nb_hidden_layers)):
        model = MLP()
        for n in range(nb_hidden_layers[i]):
            if n == 0:
                model.add_layer(nodes_per_layers[i][n], feat_size=nb_feat, activation='relu')
            else:
                model.add_layer(nodes_per_layers[i][n], activation='relu')
            #model.BatchNormalization()
        model.add_layer(nb_nodes=2, activation='softmax')
        model.compile(optimizer=optimizer, loss="cross_entropy")
        Kfold_acc, Kfold_time = model.Kfold_simulation(X, y, Kfold=10, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCH, l=l)
        history_kfold[nb_hidden_layers[i]] = Kfold_acc
        times_kfold[nb_hidden_layers[i]] = np.mean(Kfold_time)

    tunning_plot(history_kfold, times_kfold, title='k-fold accuracy related to the number of hidden layers', xlabel='Number of hidden layers')

def batch_size_tunning(X, y, BATCH_SIZES, EPOCH, optimizer='minibatch', l=0.1):
    history_kfold = {}
    times_kfold = {}
    for i in range(len(BATCH_SIZES)):
        model = MLP()
        model.add_layer(64, feat_size=nb_feat, activation='relu')
        #model.BatchNormalization()
        model.add_layer(32, activation='relu')
        #model.BatchNormalization()
        #model.add_layer(16, activation='relu')
        #model.BatchNormalization()
        model.add_layer(2, activation='softmax')
        model.compile(optimizer, loss='cross_entropy')
        Kfold_acc, Kfold_time = model.Kfold_simulation(X, y, Kfold=10, BATCH_SIZE=BATCH_SIZES[i], EPOCHS=EPOCH, l=l)
        history_kfold[BATCH_SIZES[i]] = Kfold_acc
        times_kfold[BATCH_SIZES[i]] = np.mean(Kfold_time)

    tunning_plot(history_kfold, times_kfold, title='k-fold accuracy related to the batch size', xlabel='Batch-size')

def optimizer_tunning(X,y, optimizers, BATCH_SIZE, EPOCH, l):
    history_kfold = {}
    times_kfold = {}
    for i in range(len(optimizers)):
        model = MLP()
        model.add_layer(64, feat_size=nb_feat, activation='relu')
        model.add_layer(32, activation='relu')
        #model.add_layer(16, activation='relu')
        model.add_layer(2, activation='softmax')
        model.compile(optimizer=optimizers[i], loss='cross_entropy')
        if optimizers[i] == 'sgd':
            BATCH_SIZE_i = 1
        elif optimizers[i] == 'batch':
            BATCH_SIZE_i = X.shape[0]
        elif (optimizers[i]=='adam') or (optimizers[i]=='rmsprop'):
            BATCH_SIZE_i = BATCH_SIZE
        else:
            BATCH_SIZE_i = BATCH_SIZE
        Kfold_acc, Kfold_time = model.Kfold_simulation(X, y, Kfold=10, BATCH_SIZE=BATCH_SIZE_i, EPOCHS=EPOCH, l=l)
        history_kfold[optimizers[i]] = Kfold_acc
        times_kfold[optimizers[i]] = np.mean(Kfold_time)

    tunning_plot(history_kfold, times_kfold, title='k-fold accuracy related to the optimizers', xlabel='Optimizer')

def output_layer_tunning(X, y, output_act_functions, BATCH_SIZE, EPOCH, optimizer='minibatch', l=0.1):
    history_kfold = {}
    times_kfold = {}
    for i in range(len(output_act_functions)):
        model = MLP()
        model.add_layer(64, feat_size=nb_feat, activation='relu')
        model.add_layer(32, activation='relu')
        #model.add_layer(16, activation='relu')
        if output_act_functions[i] == 'softmax':
            nodes = 2
            loss = 'cross_entropy'
        else:
            nodes = 1
            loss = 'binary_cross_entropy'
        model.add_layer(nodes, activation=output_act_functions[i])
        model.compile(optimizer=optimizer, loss=loss)
        Kfold_acc, Kfold_time = model.Kfold_simulation(X, y, Kfold=10, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCH, l=l)
        history_kfold[output_act_functions[i]] = Kfold_acc
        times_kfold[output_act_functions[i]] = np.mean(Kfold_time)

    tunning_plot(history_kfold, times_kfold,title='k-fold accuracy related to the output layer activation', xlabel='Output layer activation')

def epochs_tunning(X, y, EPOCHS, BATCH_SIZE, optimizer='minibatch', l=0.01):
    history_kfold = {}
    times_kfold = {}
    for i in range(len(EPOCHS)):
        model = MLP()
        model.add_layer(64, feat_size=nb_feat, activation='relu')
        model.add_layer(32, activation='relu')
        model.add_layer(2, activation='softmax')
        model.compile(optimizer=optimizer, loss='cross_entropy')
        Kfold_acc, Kfold_time = model.Kfold_simulation(X, y, Kfold=10, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS[i], l=l)
        history_kfold[EPOCHS[i]] = Kfold_acc
        times_kfold[EPOCHS[i]] = np.mean(Kfold_time)

    tunning_plot(history_kfold, times_kfold, title='k-fold accuracy related to the number of epochs', xlabel='Number of epochs')


def l_rate_tunning(X, y, l_rates, EPOCH, BATCH_SIZE, optimizer='minibatch'):
    history_kfold = {}
    times_kfold = {}
    for i in range(len(l_rates)):
        model = MLP()
        model.add_layer(64, feat_size=nb_feat, activation='relu')
        model.add_layer(32, activation='relu')
        model.add_layer(2, activation='softmax')
        model.compile(optimizer=optimizer, loss='cross_entropy')
        Kfold_acc, Kfold_time = model.Kfold_simulation(X, y, Kfold=10, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCH, l=l_rates[i])
        history_kfold[l_rates[i]] = Kfold_acc
        times_kfold[l_rates[i]] = np.mean(Kfold_time)
    tunning_plot(history_kfold, times_kfold, title='k-fold accuracy related to the learning rate', xlabel='Learning rate')

def act_function_tunning(X, y, act_functions, EPOCH, BATCH_SIZE, optimizer='minibatch', l=0.1):
    history_kfold = {}
    times_kfold = {}
    for i in range(len(act_functions)):
        model = MLP()
        model.add_layer(64, feat_size=nb_feat, activation=act_functions[i])
        model.add_layer(32, activation=act_functions[i])
        model.add_layer(2, activation='softmax')
        model.compile(optimizer=optimizer, loss='cross_entropy')
        Kfold_acc, Kfold_time = model.Kfold_simulation(X, y, Kfold=10, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCH, l=l)
        history_kfold[act_functions[i]] = Kfold_acc
        times_kfold[act_functions[i]] = np.mean(Kfold_time)

    tunning_plot(history_kfold, times_kfold , title='k-fold accuracy related to the activation function', xlabel='Activation function of the hidden layers')

def batch_norm_effect(X, y, act_function, EPOCH, BATCH_SIZE, optimizer='minibatch', l=0.1 ):
    model_0 = MLP()
    model_0.add_layer(64, feat_size=nb_feat, activation=act_function)
    model_0.add_layer(32, activation=act_function)
    model_0.add_layer(2, activation='softmax')
    model_0.compile(optimizer=optimizer, loss='cross_entropy')
    history_0 = model_0.fit(X, y, BATCH_SIZE, EPOCH, l=l)
    model_1 = MLP()
    model_1.add_layer(64, feat_size=nb_feat, activation=act_function)
    model_1.BatchNormalization()
    model_1.add_layer(32, activation=act_function)
    model_1.BatchNormalization()
    model_1.add_layer(2, activation='softmax')
    model_1.compile(optimizer=optimizer, loss='cross_entropy')
    history_1 = model_1.fit(X, y, BATCH_SIZE, EPOCH, l=l)

    figure, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title('Accuracy in function of epoch \n loss = {},\n optimizer = {},\n batch size = {}'.format('cross-entropy',
                                                                                                             optimizer,
                                                                                                             BATCH_SIZE))
    ax[0].plot(history_0['acc'], color='blue', label='train data')
    ax[0].plot(history_0['val_acc'], color='orange', label='val data')
    ax[0].plot(history_1['acc'], color='black',  label='train data (BatchNorm)')
    ax[0].plot(history_1['val_acc'], color='red', label='val data (BatchNorm)')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('accuracy')
    ax[1].grid()
    ax[1].plot(history_0['loss'], color='blue', label='train data')
    ax[1].plot(history_0['val_loss'], color='orange', label='val data')
    ax[1].plot(history_1['loss'], color='black', label='train data (BatchNorm)')
    ax[1].plot(history_1['val_loss'], color='red', label='val data (BatchNorm)')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('error')
    ax[1].set_title(
        'Error in function of epoch \n loss = {},\n optimizer = {},\n batch size = {}'.format('cross_entropy', optimizer,
                                                                                              BATCH_SIZE))
    ax[1].legend()
    figure.tight_layout(pad=3.0)

    plt.show()

def tunning_plot(history_kfold, times_kfold, title, xlabel, figsize=(15,10), ylabel='Accuracy'):
    """
    Plot the tunning of one hyperparameter
    :param history_kfold: dictionary with the values of k-fold accuracy for each value of the hyperparameter
    :param title: title of the plot
    :param xlabel: xlabel of the plot
    :param figsize: figsize of the plot (default (10,10))
    :param ylabel: ylabel of the plot (default 'Accuracy')
    :return: plot
    """
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.boxplot([history_kfold[key] for key in history_kfold.keys()], showfliers=False)
    plt.title(title)
    plt.xticks(np.arange(1, len(history_kfold.keys()) + 1), [key for key in history_kfold.keys()])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplot(1,2,2)
    plt.bar(np.arange(1, len(history_kfold.keys()) + 1), times_kfold.values(), tick_label=[key for key in times_kfold.keys()], width=0.4, edgecolor='black')
    plt.xticks(np.arange(1, len(history_kfold.keys()) + 1), [key for key in history_kfold.keys()])
    plt.title('k-fold time per hyperparameter value')
    plt.xlabel(xlabel)
    plt.ylabel('Time [s]')



if __name__ == "__main__":
    #  ------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    data = pd.read_csv('../wdbc.csv')
    data.columns = [i for i in range(32)]

    d = {'B': 0, 'M': 1}
    label_col = data[1].replace(d)

    data.pop(1)
    data.pop(0)
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
    X_test = data[nb_train:, :nb_feat]

    Y_train = data[:nb_train, nb_feat].astype('int')
    Y_test = data[nb_train:, nb_feat].astype('int')

    #MinMax normalization
    MM = MinMax()
    X_train = MM.fit_transform(X_train)
    X_test = MM.transform(X_test)

    # -----------------------------------------
    # K-fold cross-validation
    # -----------------------------------------
    # Hyperparameters tunning
    EPOCHS = [50, 100, 150, 200, 250, 300, 400, 500, 1000, 3000]
    nb_hidden_layers = [1, 2, 3, 4, 5]
    nodes_per_layers = [[64], [64, 32], [64, 32, 16], [128, 64, 32, 16], [128, 64, 32, 16, 8]]
    BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]
    l_rates = [0.1, 0.01, 0.001, 0.0001]
    optimizers = ["sgd", "batch", "minibatch", "rmsprop", "adam"]
    output_act_functions = ['tanh', 'sigmoid', 'softmax']
    loss_functions = ['binary_cross_entropy', 'mse', 'abs']
    hidden_layers_act = ['relu', 'tanh']

    print('---'*10)
    print("K-fold Cross-Validation")
    M = MinMax()
    X = M.fit_transform(data[:,:-1])
    y = data[:,-1]

    #Values of default hyperparameters during tuning
    EPOCH_TEST = 125
    l_rate_TEST = 0.01
    optimizer_TEST = 'minibatch'
    BATCH_SIZE_TEST = 8

    #hidden_layers_tunning(X, y, nb_hidden_layers, nodes_per_layers, BATCH_SIZE=BATCH_SIZE_TEST, EPOCH=EPOCH_TEST, optimizer=optimizer_TEST)
    #batch_size_tunning(X, y, BATCH_SIZES, EPOCH=EPOCH_TEST, l=l_rate_TEST, optimizer=optimizer_TEST)
    #optimizer_tunning(X, y, optimizers, BATCH_SIZE=BATCH_SIZE_TEST, EPOCH=EPOCH_TEST, l=l_rate_TEST)
    #batch_norm_effect(X, y, 'relu', EPOCH_TEST, BATCH_SIZE_TEST, optimizer=optimizer_TEST, l=l_rate_TEST)
    #output_layer_tunning(X, y, output_act_functions, BATCH_SIZE=BATCH_SIZE_TEST, EPOCH=EPOCH_TEST, l=l_rate_TEST, optimizer=optimizer_TEST)
    #epochs_tunning(X, y, EPOCHS, BATCH_SIZE=8)
    #l_rate_tunning(X, y, l_rates, EPOCH=EPOCH_TEST, BATCH_SIZE=BATCH_SIZE_TEST, optimizer=optimizer_TEST)
    #act_function_tunning(X, y, hidden_layers_act, EPOCH=EPOCH_TEST, BATCH_SIZE=BATCH_SIZE_TEST, optimizer=optimizer_TEST, l=l_rate_TEST)
    model = MLP()
    model.add_layer(nb_nodes=64, feat_size=nb_feat, activation='relu')
    model.add_layer(nb_nodes=32, activation='relu')
    model.add_layer(nb_nodes=2, activation='softmax')
    model.compile(optimizer_TEST, loss='cross_entropy')
    #history = model.fit(X_train, Y_train, BATCH_SIZE_TEST, EPOCH_TEST, l=l_rate_TEST)
    #model.training_curve()
    acc, loss = model.Kfold_simulation(X, y, Kfold=10, BATCH_SIZE=8, EPOCHS=125, l=0.01)
    print('10-fold accuracy : {} +-{}'.format(np.round(np.mean(acc), 4), np.round(np.std(acc), 4)))
