import numpy as np
from scipy import stats


class kNN_classifier(object):
    """
    A k Nearest Neighbour with L2 distance

    """

    def __init__(self):
        pass


    def train(self, X, Y):
        """
        :param X: training data
        :param Y: training labels
        training process memorize the data and labels, without returning anything
        """

        self.X_train = X
        self.Y_train = Y


    def compute_distance_2loops(self, X):
        """
        Compute the distance between test images in X and training images in self.X_train by using nested loops

        :param X: a numpy array of (num_test, D)
        :return dists: a numpy array of (num_test, num_train). dists[i,j] is the L2 distance between ith testing images
        and jth training images.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test,num_train), dtype='float')
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = np.linalg.norm((X[i,:]-self.X_train[j,:]))

        return dists


    def compute_distance_1loop(self, X):
        """
        Compute the distance between test images in X and training images in self.X_train by using single loop

        :param X: a numpy array of (num_test, D)
        :return dists: a numpy array of (num_test, num_train). dists[i,j] is the L2 distance between ith testing images
        and jth training images.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train), dtype = 'float')
        for i in range(num_test):
            dists_i = np.linalg.norm((self.X_train - X[i,:]), axis = 1)
            dists[i,:] += dists_i

        return dists


    def compute_distance_0loop(self, X):
        """
        Compute the distance between test images in X and training images in self.X_train using no loops at all

        :param X: a numpy array of (num_test, D)
        :return dists: a numpy array of (num_test, num_train). dists[i,j] is the L2 distance between ith testing images
        and jth training images.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train), dtype = 'float')

        mat_mul = np.matmul(X, self.X_train.transpose())

        X_norm = ((np.linalg.norm(X, axis = 1))**2)[np.newaxis].transpose()
        Xtr_norm = ((np.linalg.norm(self.X_train.transpose(), axis = 0))**2)[np.newaxis]

        print(mat_mul.shape, X_norm.shape, Xtr_norm.shape)
        dists = - 2*mat_mul + X_norm
        dists += Xtr_norm
        dists = np.sqrt(dists)

        return dists



    def predict_labels(self, dists, k=1):
        """
        Predict the classes for all test images, based on the input value k and distance matrix

        :param dists: a distance matrix
        :param k: hyperparameter for k Nearest Neighbour classifier
        :return y: a numpy array of shape (num_test, )
        """
        num_test = dists.shape[0]
        dists_index = np.argsort(dists, axis = 1)[:, :k]
        dists_classes = self.Y_train[np.concatenate(dists_index)].reshape(num_test, -1)
        y, _ = stats.mode(dists_classes, axis = 1)

        return np.squeeze(y)



