import numpy as np
from random import shuffle

def svm_loss_naive(W, X, Y, reg):
    """
    Naive implementation of SVM loss function using loops

    The images have D dimension, C different classes, totally N samples

    :param W: A numpy array of shape (D, C) containing weights
    :param X: A numpy array of shape (N, D) containing a minibatch of images
    :param Y: A numpy array of shape (N, ) containing class lables for X
    :param reg: regularization strength

    :return loss: single float  value
    :return dW: A numpy array of (D, C) containing gradients of loss function on weights
    """
    #initialize gradients as zero
    dW = np.zeros(W.shape)

    #compute loss and gradient
    num_classes = W.shape[1]
    num_samples = X.shape[0]
    loss = 0.0
    for i in range(num_samples):
        diff_count = 0
        scores = X[i].dot(W)
        correct_score = scores[Y[i]]
        for j in range(num_classes):
            if j == Y[i]:
                continue
            dW[:, j] += X[i,:]
            margin = scores[j] - correct_score + 1
            if margin > 0:
                loss += margin
                diff_count += 1

        dW[:, Y[i]] -= diff_count*X[i,:]

    loss /= num_samples
    loss += reg*np.sum(W*W)

    dW /= num_samples
    dW += 2*reg*W

    return loss, dW


def svm_loss_vectorized(W, X, Y, reg):
    """
    SVM loss function in vectorized implementation.

    Inputs and outputs are the same as the function above
    """

    ##delete list for safety reason
    try:
        del list
    except:
        pass

    dW = np.zeros(W.shape)
    loss = 0.0

    num_classes = W.shape[1]
    num_samples = X.shape[0]

    ##compute loss
    scores = X.dot(W)
    correct_scores = scores[np.arange(num_samples),Y]
    margin = scores - correct_scores[:, np.newaxis] + 1.0#perform broadcasting subtraction
    margin[np.arange(num_samples), Y] = 0  ##subtract 1 from correct class entry
    mask = margin > 0   ##a mask for thresholding
    margin_matrix = mask*margin   ##max out loss that are too small, the final loss matrix
    loss = np.sum(margin_matrix)

    ##compute gradient
    ##a mask for correct class entry in score matrix
    new_mask = np.zeros(mask.shape)
    new_mask[np.arange(num_samples), Y] += 1

    dW += (mask.transpose().dot(X)).transpose()   ##add gradient to incorrect class entry

    diff_count = np.sum(mask, axis=1)             ##diff_count of shape (N,)
    correct_entry_weights = (X*(diff_count[:,np.newaxis])).transpose()
    dW += correct_entry_weights.dot(new_mask)     ##add gradients to correct class entry


    loss /= num_samples
    loss += reg*np.sum(W*W)

    dW /= num_samples
    dW += 2*reg*W

    return loss, dW

