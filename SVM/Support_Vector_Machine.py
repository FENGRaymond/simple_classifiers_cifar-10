import numpy as np

class SVM(object):
    """
    A class for Support Vector Machine



    """

    def __init__(self):
        self.W = None


    def train(self, X, Y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):



