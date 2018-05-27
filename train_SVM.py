import random
import numpy as np
import matplotlib.pyplot as plt
from dataset.read_cifar10 import get_all_data
from SVM.loss_function import *
from SVM.gradient_check import *
import time
from SVM.Support_Vector_Machine import Linear_SVM


##Delete previous data if any, for safety reason.
try:
    del X_train, Y_train
    del X_test, Y_test
    print('Clear previously loaded data')
except:
    pass

X_train, Y_train, X_test, Y_test = get_all_data()

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

##Split the data into train, val, and test sets. In addition we will
##create a small development set as a subset of the training data;
##we can use this for development so our code runs faster
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500


##Our validation set will be num_validation points from the original training set
mask = range(num_training, num_training+num_validation)
X_val = X_train[mask]
Y_val = Y_train[mask]

##Our training set will be num_training points from the original training set
mask = range(num_training)
X_train = X_train[mask]
Y_train = Y_train[mask]

##Our development set is a subset of the training set
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
Y_dev = Y_train[mask]

##We use the first num_test points of the original set as our test set
mask = range(num_test)
X_test = X_test[mask]
Y_test = Y_test[mask]

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# As a sanity check, print out the shapes of the data
print('Training data shape: ', X_train.shape)
print('Validation data shape: ', X_val.shape)
print('Test data shape: ', X_test.shape)
print('dev data shape: ', X_dev.shape)

##Preprocessing: subtract mean image
mean_image = np.mean(X_train, axis = 0)
# print(mean_image[:10])
# plt.figure(figsize=(4,4))
# plt.imshow(mean_image.reshape((32,32,3)).astype('uint8'))
# plt.show()

X_train -= mean_image
X_val -= mean_image
X_dev -= mean_image
X_test -= mean_image

X_train = np.hstack([X_train, np.ones((X_train.shape[0],1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0],1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0],1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0],1))])

print(X_train.shape, X_val.shape, X_dev.shape, X_test.shape)

##Evaluate the naive implementation of loss function
W = np.random.rand(3073, 10)*0.0001

loss, grad = svm_loss_naive(W, X_dev, Y_dev, 0.0)
f = lambda w: svm_loss_naive(w, X_dev, Y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad)

loss, grad = svm_loss_naive(W, X_dev, Y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, Y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad)


##Compare execution time between naive implementation and vectorized implementation
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print("Naive loss: %e computed in %fs" % (loss_naive, toc-tic))

tic = time.time()
loss_vec, grad_vec = svm_loss_vectorized(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print("Vectorized loss: %e computed in %fs" % (loss_vec, toc-tic))

print("difference: %f" % (loss_naive - loss_vec))

svm = Linear_SVM()
tic = time.time()
loss_hist = svm.train(X_train, Y_train, learning_rate=1e-7, reg=2.5e5,
                      num_iters=1500, verbose=True)
toc = time.time()
print("Training SVM took %fs" % (toc-tic))
plt.plot(loss_hist)
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()

Y_train_pred = svm.predict(X_train)
print("Training accuracy : %f" % (np.mean(Y_train_pred == Y_train), ))
Y_val_pred = svm.predict(X_val)
print("Validation accuracy: %f" % (np.mean(Y_val == Y_val_pred), ))

learning_rates = [5e-8, 2e-7]
regularization_strengths = [2.5e4, 5e4]

results = {}
best_val = -1
best_svm = None

for lr in np.linspace(learning_rates[0], learning_rates[1], 5):
    for rs in np.linspace(regularization_strengths[0], regularization_strengths[1], 5):
        mySVM = Linear_SVM()
        mySVM.train(X_train, Y_train, learning_rate=lr, reg=rs,
                  num_iters=1500, verbose=True)
        Y_train_pred = mySVM.predict(X_train)
        Y_val_pred = mySVM.predict(X_val)
        train_acc = np.mean(Y_train == Y_train_pred)
        val_acc = np.mean(Y_val == Y_val_pred)
        results[(lr, rs)] = (train_acc, val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_svm = mySVM
        del mySVM

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

Y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(Y_test == Y_test_pred)
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

w = best_svm.W[:-1, :]  # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])