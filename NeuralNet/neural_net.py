import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):



    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random variables and
        biases are initialized to zeros. Weights and biases are stored in self.params,
        which is a dictionary with the following key-value pairs.

        W1: First layer weights with shape (D, H)
        b1: First layer biases with shape (H,)
        W2: Second layer weights with shape (H, C)
        b2: Second layer biases with shape (C,)


        :param input_size: The dimension D of the input data
        :param hidden_size: The number of neurons in the hidden layer
        :param output_size: The number of classes C
        :param std: standard deviation for random initialization
        """

        self.params = {}
        self.params['W1'] = std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


    def loss(self, X, Y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample
        - Y: Vector of training labels. Y[i] is the label for X[i], an integer
        in the range 0 <= Y[i] < C. This parameter is optional; if it is not passed
        then we only return scores, and if it is passed then we return the loss and gradients.
        - reg: Regularization strength


        :param X:
        :param Y:
        :param reg:
        :return: if Y is None, return a matrix scores of shape (N, C) where scores[i, c] is the
        score for class c on input X[i].

        If Y is not None, instead return a tuple of:
        - loss: Loss(data loss and regularization loss) for this batch of training samples
        - grads: Dictionary mapping parameter names to gradients of those parameters w.r.t.
        the loss function; has the same key as self.params
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        ##Compute the forward pass
        scores = None
        hidden_layer = X.dot(W1) + b1
        hidden_layer_relu = hidden_layer * (hidden_layer>0) ##hidden_layer_relu of shape (N, H)

        scores = hidden_layer_relu.dot(W2) + b2 ##output_layer of shape (N, C)

        if Y is None:
            return scores


        ##Compute the loss
        loss = None
        scores = np.exp(scores)/np.sum(np.exp(scores), axis=1, keepdims=True)
        loss = -np.sum(np.log(scores[np.arange(N), Y]))

        loss /= N
        loss += reg*np.sum(W1*W1) + reg*np.sum(W2*W2)


        ##Backward pass: compute gradients
        grads = {}
        scores[range(N), Y] -= 1
        scores /= N

        grads['W2'] = np.dot(hidden_layer_relu.T, scores)  ##(H, N)*(N, C) = (H, C)
        grads['b2'] = np.sum(scores, axis=0) ##(C,)

        dhidden = np.dot(scores, W2.T)  ##(N, C)*(C, H) = (N, H)
        dhidden[hidden_layer_relu <= 0] = 0

        grads['W1'] = np.dot(X.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis=0)

        grads['W1'] += 2*reg*W1
        grads['W2'] += 2*reg*W2


        return loss, grads


    def train(self, X, Y, X_val, Y_val, learning_rate=1e-3,
              learning_rate_decay=0.95, reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """

        :param X: A numpy matrix of shape (N, D) containing N training images
        :param Y: A numpy array of shape (N,) containing labels for X
        :param X_val: A numpy matrix of shape (N_val, D) containing N_val validation images
        :param Y_val: A numpy array of shape (N_val,) containing labels for X_val
        :param learning_rate: learning rate for optimization
        :param learning_rate_decay: a parameter to decay the learning rate after each epoch
        :param reg: regularization strength
        :param num_iters: number of steps to train
        :param batch_size: number of training images to use in each step
        :param verbose: if True, print training progress
        :return:
        """

        num_train = X.shape[0]
        iterations_per_epoch = max(int(num_train/batch_size), 1)

        ##Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            Y_batch = None

            ##Mini batch without replacement helps converge faster
            if batch_size > num_train:
                mask = np.random.choice(num_train, batch_size, replace=True)
            else:
                mask = np.random.choice(num_train, batch_size, replace=False)\


            X_batch = X[mask]
            Y_batch = Y[mask]

            loss, grads = self.loss(X_batch, Y=Y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] += -learning_rate*grads['W1']
            self.params['b1'] += -learning_rate*grads['b1']
            self.params['W2'] += -learning_rate*grads['W2']
            self.params['b2'] += -learning_rate*grads['b2']

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                ##Check accuracy
                train_acc = (self.predict(X_batch) == Y_batch).mean()
                val_acc = (self.predict(X_val) == Y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                ##Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            "loss_history": loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history,
        }


    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict input images.
        For each image we predict scores for each classes, and assign the image to
        the class with the highest score.

        :param X: A numpy matrix of shape (N, D) containing N testing images
        :return: A numpy array of shape (N,) containing predicted class labels for X.
        """

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        Y_pred = None

        hidden_layer = X.dot(W1) + b1
        hidden_layer_relu = np.maximum(hidden_layer, 0)

        output_layer = hidden_layer_relu.dot(W2) + b2
        Y_pred = np.argmax(output_layer, axis=1)

        return Y_pred

