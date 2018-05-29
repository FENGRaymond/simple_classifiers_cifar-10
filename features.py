import random
import numpy as np
from dataset.read_cifar10 import *
import matplotlib.pyplot as plt
from Features.extract_features import color_histogram_hsv, hog_feature, extract_features
from SVM.Support_Vector_Machine import Linear_SVM
from NeuralNet.neural_net import TwoLayerNet

def get_CIFAR10_data_feature(num_training=49000, num_valiadtion=1000, num_test=1000, num_dev=500):
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

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

try:
    del X_train, Y_train
    del X_test, Y_test
    print("Clear previously loaded data")
except:
    pass

X_train, Y_train, X_val, Y_val, X_test, Y_test = get_CIFAR10_data_feature()

num_color_bins = 10
features_fns = [hog_feature, lambda img:color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, features_fns, verbose=True)
X_val_feats = extract_features(X_val, features_fns)
X_test_feats = extract_features(X_test, features_fns)

mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
# X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
# X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])


##USING SVM
# learning_rates = [1e-9, 1e-8, 1e-7]
# regularization_strengths = [5e4, 5e5, 5e6]
#
# results = {}
# best_val = -1
# best_svm = None
#
# for learning_rate in learning_rates:
#     for regularization_strength in regularization_strengths:
#         SVM = Linear_SVM()
#         SVM.train(X_train_feats, Y_train,
#                   learning_rate=learning_rate, reg=regularization_strength,
#                   num_iters=1500, batch_size=200, verbose=True)
#         Y_train_pred = SVM.predict(X_train_feats)
#         Y_val_pred = SVM.predict(X_val_feats)
#         train_acc = np.mean(Y_train == Y_train_pred)
#         val_acc = np.mean(Y_val == Y_val_pred)
#         results[(learning_rate, regularization_strength)] = (train_acc, val_acc)
#         if val_acc > best_val:
#             best_val = val_acc
#             best_svm = SVM
#         del SVM
#
# for lr, reg in sorted(results):
#     train_accuracy, val_accuracy = results[(lr, reg)]
#     print("lr %e reg %e train accuracy: %f val accuracy: %f" %
#           (lr, reg, train_accuracy, val_accuracy))
#
# print("best validation accuracy achieved: %f" % best_val)
#
# Y_test_pred = best_svm.predict(X_test_feats)
# test_accuracy = np.mean(Y_test == Y_test_pred)
# print(test_accuracy)
#
# examples_per_class = 8
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# for cls, cls_name in enumerate(classes):
#     idxs = np.where((Y_test != cls) & (Y_test_pred == cls))[0]
#     idxs = np.random.choice(idxs, examples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
#         plt.imshow(X_test[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls_name)
# plt.show()

##USING NEURAL NET
input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

results = {}
best_net = None
best_val = 0

learning_rates = [0.9, 1, 1.1]
regs = [5e-5, 1e-4, 5e-4]

for lr in learning_rates:
    for reg in regs:
        net = TwoLayerNet(input_dim, hidden_dim, num_classes)
        print(lr, reg)
        stats = net.train(X_train_feats, Y_train, X_val_feats, Y_val,
                          learning_rate=lr, reg=reg, num_iters=2000, batch_size=200, verbose=True)
        Y_train_pred = net.predict(X_train_feats)
        Y_val_pred = net.predict(X_val_feats)
        train_acc = np.mean(Y_train == Y_train_pred)
        val_acc = np.mean(Y_val == Y_val_pred)
        results[(lr, reg)] = (train_acc, val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_net = net
        del net

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print("lr %e reg %e train accuracy: %f val accuracy: %f" %
          (lr, reg, train_accuracy, val_accuracy))

print("best validation accuracy achieved: %f" % best_val)

Y_test_pred = best_net.predict(X_test_feats)
test_accuracy = np.mean(Y_test == Y_test_pred)
print(test_accuracy)