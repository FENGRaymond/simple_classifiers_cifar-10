import numpy as np
import matplotlib.pyplot as plt
from NeuralNet.cnn import *
from FullyConnectedNet import get_CIFAR10_data, rel_error
from SVM.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from NeuralNet.layers import *
from NeuralNet.fast_layers import *
from solver import Solver
from scipy.misc import imread, imresize

def imshow_noax(img, normalize=True):
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0*(img-img_min)/(img_max-img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


if __name__ == '__main__':
    data = get_CIFAR10_data()

    num_train = 5000
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['Y_train'][:num_train],
        'X_val': data['X_val'][:num_train],
        'y_val': data['Y_val'][:num_train],
    }


    ##--------------------------------Test convolution naive forward: pass!----------------------------
    # x_shape = (2,3,4,4)
    # w_shape = (3,3,4,4)
    # x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    # w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    # b = np.linspace(-0.1, 0.2, num=3)
    #
    # conv_param = {'stride': 2, 'pad': 1}
    # out, _ = conv_forward_naive(x, w, b, conv_param)
    # correct_out = np.array([[[[-0.08759809, -0.10987781],
    #                        [-0.18387192, -0.2109216 ]],
    #                       [[ 0.21027089,  0.21661097],
    #                        [ 0.22847626,  0.23004637]],
    #                       [[ 0.50813986,  0.54309974],
    #                        [ 0.64082444,  0.67101435]]],
    #                      [[[-0.98053589, -1.03143541],
    #                        [-1.19128892, -1.24695841]],
    #                       [[ 0.69108355,  0.66880383],
    #                        [ 0.59480972,  0.56776003]],
    #                       [[ 2.36270298,  2.36904306],
    #                        [ 2.38090835,  2.38247847]]]])
    #
    # print('Testing conv_forward_naive')
    # print('difference: ', rel_error(out, correct_out))
    #
    #
    # ##
    # kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
    # d = kitten.shape[1] - kitten.shape[0]
    # kitten_cropped = kitten[:, d//2:-d//2, :]
    #
    # img_size = 200
    # x = np.zeros((2,3,img_size, img_size))
    # x[0,:,:,:] = imresize(puppy, (img_size, img_size)).transpose(2,0,1)
    # x[1,:,:,:] = imresize(kitten_cropped, (img_size, img_size)).transpose(2,0,1)
    #
    # w = np.zeros((2,3,3,3))
    #
    # ##First filter converts RGB to grayscale
    # w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
    # w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
    # w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
    #
    # ##Second filter detects edges in blue channel
    # w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    # b = np.array([0, 128])
    #
    # out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})
    #
    #
    # plt.subplot(2,3,1)
    # imshow_noax(puppy, normalize=False)
    # plt.title('Original image')
    # plt.subplot(2,3,2)
    # imshow_noax(out[0,0])
    # plt.title('Grayscale')
    # plt.subplot(2,3,3)
    # imshow_noax(out[0,1])
    # plt.title('Edges')
    # plt.subplot(2,3,4)
    # imshow_noax(kitten_cropped, normalize=False)
    # plt.subplot(2,3,5)
    # imshow_noax(out[1,0])
    # plt.subplot(2,3,6)
    # imshow_noax(out[1,1])
    # plt.show()


    ##-------------------------------------test naive backward: pass!-------------------------------------
    # np.random.seed(231)
    # x = np.random.randn(4, 3, 5, 5)
    # w = np.random.randn(2, 3, 3, 3)
    # b = np.random.randn(2,)
    # dout = np.random.randn(4, 2, 5, 5)
    # conv_param = {'stride': 1, 'pad': 1}
    #
    # dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
    # dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
    # db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)
    #
    # out, cache = conv_forward_naive(x, w, b, conv_param)
    # dx, dw, db = conv_backward_naive(dout, cache)
    #
    # print('Testing conv_backward_naive function')
    # print('dx error: ', rel_error(dx, dx_num))
    # print('dw error: ', rel_error(dw, dw_num))
    # print('db error: ', rel_error(db, db_num))

    ##----------------------------------test naive max-pooling: pass!-------------------------------------------
    # x_shape = (2,3,4,4)
    # x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
    # pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
    #
    # out, _ = max_pool_forward_naive(x, pool_param)
    #
    # correct_out = np.array([[[[-0.26315789, -0.24842105],
    #                           [-0.20421053, -0.18947368]],
    #                          [[-0.14526316, -0.13052632],
    #                           [-0.08631579, -0.07157895]],
    #                          [[-0.02736842, -0.01263158],
    #                           [0.03157895, 0.04631579]]],
    #                         [[[0.09052632, 0.10526316],
    #                           [0.14947368, 0.16421053]],
    #                          [[0.20842105, 0.22315789],
    #                           [0.26736842, 0.28210526]],
    #                          [[0.32631579, 0.34105263],
    #                           [0.38526316, 0.4]]]])
    #
    # print('Testing max_pool_forward_naive function:')
    # print('difference: ', rel_error(out, correct_out))


    # np.random.seed(231)
    # x = np.random.randn(3,2,8,8)
    # dout = np.random.randn(3,2,4,4)
    # pool_param = {'pool_height':2, 'pool_width':2, 'stride':2}
    #
    # dx_num = eval_numerical_gradient_array(lambda x:max_pool_forward_naive(x, pool_param)[0], x, dout)
    #
    # out, cache = max_pool_forward_naive(x, pool_param)
    # dx = max_pool_backward_naive(dout, cache)
    #
    # print('Testing max_pool_bacward_naive function: ')
    # print('dx error: ', rel_error(dx, dx_num))

    # TODO: fast layers implementation