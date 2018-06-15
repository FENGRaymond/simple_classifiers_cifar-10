import numpy as np
import matplotlib.pyplot as plt
from NeuralNet.layers import *
from NeuralNet.layer_utils import *

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


class FullyConnectedNet(object):
    """
    A fully-connected nueral network with an arbitrary number of hidden layers,
    ReLU nonlinerities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L-1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block
    repeated L-1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the self.params
    dictionary and will be learned using the Slover class.

    """


    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """

        :param hidden_dims: A list of integers giving the size of each hidden layer.
        :param input_dim: An integer giving the size of the input.
        :param num_classes: An integer giving the number of classes to classify.
        :param dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        :param normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        :param reg: Scalar giving L2 regularization strength.
        :param weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        :param dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        :param seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_normalization = normalization != None
        self.use_dropoout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        self.params['W1'] = weight_scale*np.random.randn(input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])

        hidden_dims.append(num_classes)

        for i in range(len(hidden_dims)-1):
            w_name = 'W' + str(i+2)
            b_name = 'b' + str(i+2)
            self.params[w_name] = weight_scale*np.random.randn(hidden_dims[i], hidden_dims[i+1])
            self.params[b_name] = np.zeros(hidden_dims[i+1])

        if self.use_normalization:
            for i in range(self.num_layers - 1):
                self.params['gamma%d' % (i+1)] = np.ones(hidden_dims[i])
                self.params['beta%d' % (i+1)] = np.zeros(hidden_dims[i])


        self.dropout_param = {}
        if self.use_dropoout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        for k,v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, Y=None):
        """
        Compute loss and gradient for the fully-connected network.

        :param X:
        :param Y:
        :return:
        """

        X = X.astype(self.dtype)
        mode = 'test' if Y is None else 'train'

        ##Set train/test mode for batchnorm params and dropout param since
        ##they behave differently during training and testing.
        if self.use_dropoout:
            self.dropout_param['mode'] = mode

        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode


        scores = None

        fc_cache = {}
        bn_cache = {}
        relu_cache = {}
        dropout_cache = {}

        ##Reshape input batch
        X = np.reshape(X, [X.shape[0], -1])

        ##Forward pass
        ##There will be L-1 number of iterations of {affine - (normalization) - relu - (dropout)}
        layer = X
        for i in range(self.num_layers-1):

            ##1 - fully connected layer
            fc_layer, fc_cache[str(i+1)] = affine_forward(layer, self.params['W%d'%(i+1)], (self.params['b%d'%(i+1)]))

            ##2 - batch normalization layer
            if self.normalization == 'batchnorm':
                fc_layer, bn_cache[str(i+1)] = batchnorm_forward(fc_layer, self.params['gamma%d' % (i+1)], self.params['beta%d' % (i+1)], self.bn_params[i])
            elif self.normalization == 'layernorm':
                fc_layer, bn_cache[str(i+1)] = layernorm_forward(fc_layer, self.params['gamma%d' % (i+1)], self.params['beta%d' % (i+1)], self.bn_params[i])

            ##3 - ReLU activation layer
            layer, relu_cache[str(i+1)] = relu_forward(fc_layer)

            ##4 - dropout layer
            if self.use_dropoout:
                layer, dropout_cache[str(i+1)] = dropout_forward(layer, self.dropout_param)

        scores, fc_cache[str(self.num_layers)] = affine_forward(layer, self.params['W%d'%(self.num_layers)], self.params['b%d'%(self.num_layers)])
        if mode == 'test':
            return scores

        ##Backpropagation
        loss, grads = 0.0, {}

        data_loss, dout = softmax_loss(scores, Y)
        reg_loss = 0.0
        for w in [self.params['W%d'%(i+1)] for i in range(self.num_layers)]:
            reg_loss += 0.5*self.reg*np.sum(w*w)
        loss = data_loss + reg_loss ##Not sure


        dx, dw, db = affine_backward(dout, fc_cache[str(self.num_layers)])
        grads['W%d'%self.num_layers] = dw + self.reg*self.params['W%d'%self.num_layers]
        grads['b%d'%self.num_layers] = db

        for j in reversed(range(self.num_layers-1)):
            ##4 - dropout
            if self.use_dropoout:
                dx = dropout_backward(dx, dropout_cache[str(j+1)])

            ##3 - relu
            dx = relu_backward(dx, relu_cache[str(j+1)])

            ##2 - batch normalization
            if self.normalization == 'batchnorm':
                dx, dgamma, dbeta = batchnorm_backward(dx, bn_cache[str(j+1)])
                grads['gamma%d'%(j+1)] = dgamma
                grads['beta%d'%(j+1)] = dbeta
            elif self.normalization == 'layernorm':
                dx, dgamma, dbeta = layernorm_backward(dx, bn_cache[str(j+1)])

            ##1 - fully connected
            dx, dw, db = affine_backward(dx, fc_cache[str(j+1)])
            grads['W%d'%(j+1)] = dw + self.reg*self.params['W%d'%(j+1)]
            grads['b%d'%(j+1)] = db


        return loss, grads


class TwoLayerNet_v2(object):

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network

        :param input_dim: An integer giving the size of the input
        :param hidden_dim: An integer giving the size of the hidden layer
        :param num_classes: An integer giving the number of classes(size of output layer)
        :param weight_scale: Scalar giving the standard deviation for random initialization of weights
        :param reg: Scalar giving L2 regularization strength
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale*np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale*np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, Y=None):
        """
        Compute the loss and gradient for a minibatch of data

        :param X: Array of input data of shape (N, d_1, ..., d_k)
        :param Y: Array of labels, of shape (N,). Y[i] gives the label for X[i]
        :return:
                if Y is None, then run the test-time forward pass of module and return
                - scores: Array of shape (N, C) giving classification scores, where
                scores[i, c] is the classification score for X[i] and class c.

                if Y is not None, then run a training-time forward and backward pass and
                return a tuple of:
                - loss: Scalar value giving the loss
                - grads: Dictionary with the same keys as self.params, mapping parameter
                names to gradients of the loss w.r.t. those parameters
        """
        N = X.shape[0]
        X = X.reshape(X.shape[0], -1)

        scores = None
        l1_out, l1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, l2_cache = affine_forward(l1_out, self.params['W2'], self.params['b2'])

        if Y is None:
            return scores

        loss, grads = 0.0, {}

        ##Calcualte loss
        data_loss, dout = softmax_loss(scores, Y)

        reg_loss = 0.0
        for W in [self.params[param] for param in self.params if param[0]=='W']:
            reg_loss += 0.5*self.reg*np.sum(W*W)

        loss = data_loss + reg_loss
        ##Calculate gradients
        dx2, dw2, db2 = affine_backward(dout, l2_cache)
        dx1, dw1, db1 = affine_relu_backward(dx2, l1_cache)

        grads['W1'] = dw1 + self.reg*self.params['W1']
        grads['b1'] = db1
        grads['W2'] = dw2 + self.reg*self.params['W2']
        grads['b2'] = db2

        return loss, grads

##6/16/2018