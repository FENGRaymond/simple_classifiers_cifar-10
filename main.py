import numpy as np
import random
import matplotlib.pyplot as plt
from dataset.read_cifar10 import get_all_data
from kNN.kNN_class import kNN_classifier

def time_function(f, *args):
    """
    Call a function f with args and return the time in seconds that it took

    :param f: a function
    :param args: arguments for function f
    :return: execution time
    """

    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

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

# ##Visualize examples from the dataset
# ##We show 7 training examples from each class
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(Y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace = False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i*num_classes + y +1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

##Subsample the data to speed up
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
Y_train = Y_train[mask]

num_test = 5000
mask = list(range(num_test))
X_test = X_test[mask]
Y_test = Y_test[mask]

#reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

##Create a kNN classifier instance.
classifier = kNN_classifier()
classifier.train(X_train, Y_train)

dists = classifier.compute_distance_0loop(X_test)
print(dists.shape)

# ##Visualize the distance matrix
# plt.imshow(dists, interpolation='none')
# plt.show()

Y_test_pred = classifier.predict_labels(dists, k=1)
num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct)/num_test
print('Got %d/%d correct => accuracy: %f' % (num_correct, num_test, accuracy))


##Cross validation with n folds
num_folds = 5
k_choices = [1,3,5,8,10,12,15,20,50,100]

x_train_folds = np.split(X_train, indices_or_sections=num_folds)
y_train_folds = np.split(Y_train, indices_or_sections=num_folds)

##A dictionary to store accuracies for different k
k_to_accuracies = {}
for k in k_choices:
    k_to_accuracies[k] = []
    for fold in range(num_folds):
        Xtr_cv = np.concatenate(([x_train_folds[i] for i in range(5) if i!= fold]))
        Ytr_cv = np.concatenate(([y_train_folds[i] for i in range(5) if i!= fold]))
        ##print(Xtr_cv.shape, Ytr_cv.shape)
        Xte_cv = x_train_folds[fold]
        Yte_cv = y_train_folds[fold]
        classifier2 = kNN_classifier()
        classifier2.train(Xtr_cv, Ytr_cv)
        dist_cv = classifier2.compute_distance_0loop(Xte_cv)
        Yte_pred = classifier2.predict_labels(dist_cv, k=k)
        num_correct = np.sum(Yte_pred == Yte_cv)
        accuracy = num_correct/1000.0
        k_to_accuracies[k].append(accuracy)


for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

##Plot the raw observation
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k]*len(accuracies), accuracies)

##plot the trend line with error bars hat correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

