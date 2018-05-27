import random
import numpy as np
from dataset.read_cifar10 import get_all_data
import matplotlib.pyplot as plt
from SVM.softmax import *
import time
from SVM.gradient_check import grad_check_sparse
from SVM.Support_Vector_Machine import *

def get_CIFAR10_data(num_training=49000, num_valiadtion=1000, num_test=1000, num_dev=500):
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
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]

    ##Reshape the images into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    ##Subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    ##Add bias dimension and transform into columns
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    X_dev = np.hstack((X_dev, np.ones((X_dev.shape[0], 1))))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev

##Clear variables to prevent loading multiple times
try:
    del X_train, Y_train
    del X_test, Y_test
    print("Cleared previous data")
except:
    pass


X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev = get_CIFAR10_data()

W = np.random.rand(3073, 10) * 0.0001


"""
Experiment results show that naive and vectorized implementation have no difference, 
but with the latter 30 times faster than the former
"""
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print("Naive loss: %e computed in %fs" % (loss_naive, toc-tic))

tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print("Vectorized loss: %e computed in %fs" % (loss_vectorized, toc-tic))

grad_diff = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print("Loss difference: %f" % np.abs(loss_naive-loss_vectorized))
print("grad difference: %f" % grad_diff)


results = {}
best_val = -1
best_softmax = None
best_loss_hist = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]

for lr in learning_rates:
    for rs in regularization_strengths:
        mySoftmax = Softmax()
        loss_hist = mySoftmax.train(X_train, Y_train, learning_rate=lr, reg=rs,
                  num_iters=1500, verbose=True)
        Y_train_pred = mySoftmax.predict(X_train)
        Y_val_pred = mySoftmax.predict(X_val)
        train_acc = np.mean(Y_train == Y_train_pred)
        val_acc = np.mean(Y_val == Y_val_pred)
        results[(lr, rs)] = (train_acc, val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_softmax = mySoftmax
            best_loss_hist = loss_hist
        del mySoftmax
plt.plot(best_loss_hist)
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()


for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print("lr %e reg %e train accuracy: %f val accuracy: %f" % (
        lr, reg, train_accuracy, val_accuracy
    ))

print("best validation accuracy achieved during cross-validation: %f" % best_val)

Y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(Y_test == Y_test_pred)
print("Softmax on raw pixels final test set accuracy: %f" % (test_accuracy))
#
# w = best_softmax.W[:-1, :]
# w = w.reshape(32,32,3,10)
#
# w_min, w_max = np.min(w), np.max(w)
#
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#
#     wimg = 255.0 * (w[:,:,:,i].squeeze() - w_min)/(w_max-w_min)
#     plt.imshow(wimg.astype('uint8'))
#     plt.axis('off')
#     plt.title(classes[i])