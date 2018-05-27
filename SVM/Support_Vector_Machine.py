import numpy as np
from SVM.loss_function import *
from SVM.softmax import *


##Superclass Linear_Classifier
class Linear_Classifier(object):
    """
    A superclass of a linear classifier

    train: train the input batch using stochastic gradient decent
    predict: based on the trained weights, predict the class labels of input images
    loss: a function to calculate loss
    """

    def __init__(self):
        self.W = None


    def train(self, X, Y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Stochastic gradient decent

        :param X: A numpy array of shape (N, D) containing training images, there are N samples in total
        :param Y: A numpy array of shape (N,) which are corresponding class labels for X
        :param learning_rate: (float) learning rate for optimization
        :param reg: (float) regularization strength
        :param num_iters: (integer) number of iterations when optimizing
        :param batch_size: (integer) number of training images to use in each iteration
        :param verbose: (boolean) If true, print progress during optimization

        :return: A list containing the value of the loss function at each optimization iteration
        """
        num_train, dim = X.shape
        num_classes = Y.max() + 1
        if self.W is None:
            ##Layzily initialize W
            self.W = 0.001 * np.random.rand(dim, num_classes)

        ##Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            random_index = np.random.choice(np.arange(num_train), batch_size, replace=False)
            X_batch = X[random_index]
            Y_batch = Y[random_index]

            loss, grad = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)

            self.W += -learning_rate*grad

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history


    def predict(self, X):
        """
        Used the trained weights of this linear classifier to predict labels for data points

        :param X: A numpy array of shape (N, D) containing testing images
        :return: A numpy array of shape (N,) containing predicted labels for the images in X.
        """
        Y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        Y_pred += np.argmax(scores, axis=1)
        return Y_pred


    def loss(self, X_batch, Y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclass will override this.

        :param X_batch: A numpy array of shape (N, D) containing a minibatch training images
        :param Y_batch: A numpy array of shape (N,) of corresponding class labels
        :param reg: (float) regularization strength
        :return:A tuple containing
                -loss as a single float
                -gradient with respect to self.W; an array of the same shape as W
        """
        #
        # try:
        #     del list
        # except:
        #     pass
        #
        # dW = np.zeros(self.W.shape)
        # loss = 0.0
        #
        # num_classes = self.W.shape[1]
        # num_samples = X_batch.shape[0]
        #
        # ##Compute loss
        # scores = X_batch.dot(self.W)
        # correct_scores = scores[np.arange(num_samples), Y_batch]
        # margin = scores - correct_scores[:, np.newaxis] + 1.0  # perform broadcasting subtraction
        # margin[np.arange(num_samples), Y_batch] = 0  ##subtract 1 from correct class entry
        # mask = margin > 0  ##a mask for thresholding
        # margin_matrix = mask * margin  ##max out loss that are too small, the final loss matrix
        # loss = np.sum(margin_matrix)
        #
        # ##Computer gradient
        # new_mask = np.zeros(mask.shape)
        # new_mask[np.arange(num_samples), Y_batch] += 1
        #
        # dW += (X_batch.T).dot(mask)
        # diff_count = np.sum(mask, axis=1)  ##diff_count of shape (N,)
        # correct_entry_weights = (X_batch * (diff_count[:, np.newaxis])).transpose()
        # dW += correct_entry_weights.dot(new_mask)  ##add gradients to correct class entry
        #
        # loss /= num_samples
        # loss += reg*np.sum(self.W*self.W)
        #
        # dW /= num_samples
        # dW += 2*reg*self.W
        #
        # return loss, dW
        pass



##Subclass Linear_SVM
class Linear_SVM(Linear_Classifier):
    """
    A subclass that uses the multiclass SVM loss function

    """
    def loss(self, X_batch, Y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, Y_batch, reg)


##Subclass Softmax
class Softmax(Linear_Classifier):
    """
    A subclass that uses the softmax + cross-entropy loss function

    """
    def loss(self, X_batch, Y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, Y_batch, reg)