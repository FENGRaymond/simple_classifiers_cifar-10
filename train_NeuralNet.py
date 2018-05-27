import numpy as np
import matplotlib.pyplot as plt
from NeuralNet.neural_net import *
from dataset.read_cifar10 import *
from vis_utils import visualize_grid
import itertools
import time

def rel_error(x, y):
    return np.max(np.abs(x-y)/(np.maximum(1e-8, np.abs(x)+np.abs(y))))

def init_toy_model(input_size, hidden_size, num_classes):
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data(num_inputs, input_size):
    np.random.seed(1)
    X = 10*np.random.randn(num_inputs, input_size)
    Y = np.array([0,1,2,2,1])
    return X, Y

def get_CIFAR10_data_DNN(num_training=49000, num_valiadtion=1000, num_test=1000, num_dev=500):
    """
    Some pre-processing of CIFAR10 data
    """

    X_train, Y_train, X_test, Y_test = get_all_data()

    ##Subsampling
    mask = list(range(num_training, num_training+num_valiadtion))
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]

    ##Reshape the images into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    ##Subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def check_network():
    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5

    X, Y = init_toy_data(num_inputs, input_size)

    net = init_toy_model(input_size, hidden_size, num_classes)
    stats = net.train(X, Y, X, Y,
                      learning_rate=1e-1, reg=5e-6,
                      num_iters=100, verbose=False)
    print("Final training loss: ", stats['loss_history'][-1])

    ##Plot the loss history
    plt.plot(stats['loss_history'])
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training loss history')
    plt.show()

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32,32,3,-1).transpose(3,0,1,2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

if __name__ == "__main__":
    try:
        del X_train, Y_train
        del X_test, Y_test
        print("Clear previously loaded data")
    except:
        pass

    X_train, Y_train, X_val, Y_val, X_test, Y_test,  = get_CIFAR10_data_DNN()
    input_size = 32*32*3
    hidden_choices = [40, 50, 60, 70, 80, 100]
    num_classes = 10

    learning_rate = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    num_iter = [1000, 2000, 3000]
    batch_size = [128, 200, 256]
    reg = [0.2, 0.25, 0.3]
    combinations = list(itertools.product(learning_rate, num_iter, batch_size, reg))

    best_net = None
    best_val_acc = 0

    for hidden_size in hidden_choices:
        for i in range(len(combinations)):
            net = TwoLayerNet(input_size, hidden_size, num_classes)

            ##Train the network
            stats = net.train(X_train, Y_train, X_val, Y_val,
                              learning_rate=combinations[i][0], learning_rate_decay=0.95,
                              num_iters=combinations[i][1], batch_size=combinations[i][2], reg=combinations[i][3],verbose=True)


            ##Predict on the validation set
            val_acc = np.mean(net.predict(X_val) == Y_val)

            if val_acc > best_val_acc:
                best_net = net
                best_val_acc = val_acc
                print("Validation accuracy: ", val_acc)

                # plt.subplot(2,1,1)
                # plt.plot(stats['loss_history'])
                # plt.xlabel('iteration')
                # plt.ylabel('loss')
                # plt.title('loss history')
                #
                # plt.subplot(2,1,2)
                # plt.plot(stats['train_acc_history'], label='train')
                # plt.plot(stats['val_acc_history'], label='val')
                # plt.title('Classification accuracy history')
                # plt.xlabel('Epoch')
                # plt.ylabel('Classification accuracy')
                # plt.legend()
                # plt.show()

            else:
                del net
