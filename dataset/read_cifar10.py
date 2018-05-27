import pickle
import numpy as np

PREFIX = 'dataset/cifar-10-batches-py/'
BATCH_TRAIN = [ PREFIX+'data_batch_'+str(x) for x in range(1,6)]
BATCH_TEST = PREFIX+'test_batch'

def unpickle(file):
    """
    Input: pickle files of different data batches
    Output: a dictionary that has the following key-value pairs
            b'batch_label' - b'...batch x of X'
            b'labels' - a list of 10000 labels
            b'data' - 10000x3072 numpy array
            b'filenames' - a list of 10000 names
    """

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        return dict

def get_all_data():
    """

    :return: X_train, Y_train, X_test, Y_test
    in
    """
    Xtr = []
    Ytr = []
    Xte = []
    Yte = []

    for pname in BATCH_TRAIN:
        dict = unpickle(pname)
        X = dict[b'data']
        Y = dict[b'labels']
        X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
        Xtr.append(X)
        Ytr.append(Y)
        print(dict[b'batch_label'].decode('ascii'))

    Xtr = np.concatenate(Xtr)
    Ytr = np.concatenate(Ytr)
    del X, Y, dict
    ##print('Xtr shape: ',Xtr.shape)
    ##print('Ytr shape: ',Ytr.shape)

    dict = unpickle(BATCH_TEST)
    X = dict[b'data']
    Y = dict[b'labels']
    X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('float')
    ##print(dict[b'batch_label'].decode('ascii'))
    Xte = X
    Yte = np.array(Y)
    ##print('Xte:', Xte.shape, Xte.dtype)
    ##print('Yte:', Yte.shape, Yte.dtype)

    return Xtr, Ytr, Xte, Yte

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_all_data()
