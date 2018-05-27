import numpy as np

def softmax_loss_naive(W, X, Y, reg):
    """
    Softmax loss function, naive implementation with loops

    :param W: A numpy array of shape (D, C) containing weights
    :param X: A numpy array of shape (N, D) containing N training images
    :param Y: A numpy array of shape (N,) containing corresponding class labels
    :param reg: (float) regularization strength
    :return:
            -loss as a single float
            -gradient w.r.t. to weights W, an array of shape (D, C)
    """
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_samples = X.shape[0]

    for i in range(num_samples):
        scores = X[i].dot(W)

        ##Calculate softmax
        scores -= np.max(scores)
        softmax_scores = np.exp(scores)/np.sum(np.exp(scores))

        ##Add cross entropy loss to variable loss
        loss -= np.log(softmax_scores[Y[i]])

        for j in range(num_classes):
            if j == Y[i]:
                dW[:, j] += X[i]*(np.exp(scores[j]))/np.sum(np.exp(scores))-X[i]
            else:
                dW[:, j] += X[i]*(np.exp(scores[j]))/np.sum(np.exp(scores))


    loss /= num_samples
    loss += reg*np.sum(W*W)

    dW /= num_samples
    dW += 2*reg*W

    return loss, dW


def softmax_loss_vectorized(W, X, Y, reg):
    """
    A vectorized implementation of softmax loss. Takes in the same inputs and return the same outputs
    """

    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_samples = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1)[:, np.newaxis]

    ##softmax_scores is of shape (N, C)
    softmax_scores = np.exp(scores)/np.sum(np.exp(scores), axis=1)[:, np.newaxis]
    loss -= np.sum(np.log(softmax_scores[np.arange(num_samples),Y]))

    ##dW is of shape (D, C) = (D,N).*(N, C)
    softmax_scores[np.arange(num_samples), Y] -= 1
    dW = (X.T).dot(softmax_scores)

    loss /= num_samples
    loss += reg*np.sum(W*W)

    dW /= num_samples
    dW += 2*reg*W


    return loss, dW
