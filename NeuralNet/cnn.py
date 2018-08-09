from builtins import object
import numpy as np

from NeuralNet.layers import *

#import error, comment out
#from NeuralNet.fast_layers import *
from NeuralNet.layer_utils import *

from ConvolutionalNetworks import CYTHON_PASS




class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params['W1'] = weight_scale*np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters, dtype=dtype)
        self.params['W2'] = weight_scale*np.random.randn(int(input_dim[1]/2)*int(input_dim[2]/2)*num_filters, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim, dtype=dtype)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes, dtype=dtype)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None


        if CYTHON_PASS:
            pass
            #conv_out, conv_cache = conv_relu_forward(X, W1, b1, conv_param)
            #pool_out, pool_cache = max_pool_forward_fast(conv_out, pool_param)
        else:
            conv_out, conv_cache = conv_relu_forward_alt(X, W1, b1, conv_param)
            pool_out, pool_cache = max_pool_forward_naive(conv_out, pool_param)

        #pool_out_flat = pool_out.flatten()

        aff_out, aff_cache = affine_relu_forward(pool_out, W2, b2)
        scores, out_cache = affine_forward(aff_out, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}

        data_loss, dout = softmax_loss(scores, y)
        reg_loss = 0.0
        for w in [self.params['W%d' % (i+1)] for i in range(3)]:
            reg_loss += 0.5*self.reg*np.sum(w*w)
        loss = data_loss + reg_loss


        ##Backpropagation
        dx_3, dw_3, db_3 = affine_backward(dout, out_cache)
        dx_2, dw_2, db_2 = affine_relu_backward(dx_3, aff_cache)
        dx_2_deflat = np.reshape(dx_2, pool_out.shape)

        if CYTHON_PASS:
            pass
            #dx_pool = max_pool_backward_fast(dx_2_deflat, pool_cache)
            #dx_1, dw_1, db_1 = conv_relu_backward(dx_pool, conv_cache)
        else:
            dx_pool = max_pool_backward_naive(dx_2_deflat, pool_cache)
            dx_1, dw_1, db_1 = conv_relu_backward_alt(dx_pool, conv_cache)

        grads['W1'] = dw_1 + self.reg*W1
        grads['b1'] = db_1
        grads['W2'] = dw_2 + self.reg*W2
        grads['b2'] = db_2
        grads['W3'] = dw_3 + self.reg*W3
        grads['b3'] = db_3

        return loss, grads
