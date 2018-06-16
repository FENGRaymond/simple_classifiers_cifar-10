import time
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet.neural_net import *
from FullyConnectedNet import get_CIFAR10_data, rel_error
from SVM.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from solver import Solver



if __name__ == '__main__':
    data = get_CIFAR10_data()

    ##-----------------------------Test dropout forward: pass!-----------------------------
    # np.random.seed(231)
    # x = np.random.randn(500,500) + 10
    # for p in [0.25, 0.4, 0.7]:
    #     out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
    #     out_test, _ = dropout_forward(x, {'mode':'test', 'p': p})
    #
    #     print('Running tests with p = ', p)
    #     print('Mean of input: ', x.mean())
    #     print('Mean of train-time output: ', out.mean())
    #     print('Mean of test-time output: ', out_test.mean())
    #     print('Fraction of train-time output set to zero: ', (out == 0).mean())
    #     print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
    #     print()


    ##----------------------------Test dropout backward: pass!----------------------------------
    # np.random.seed(231)
    # x = np.random.randn(10, 10)+10
    # dout = np.random.randn(*x.shape)
    #
    # dropout_param = {'mode':'train', 'p': 0.2, 'seed': 123}
    # out, cache = dropout_forward(x, dropout_param)
    # dx = dropout_backward(dout, cache)
    # dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)
    # print('dx relative error: ', rel_error(dx, dx_num))


    ##----------------------------Fully-connected nets with dropout: pass!---------------------------
    # np.random.seed(231)
    # N, D, H1, H2, C = 2, 15, 20, 30, 10
    # X = np.random.randn(N, D)
    # y = np.random.randint(C, size=(N,))
    #
    # for dropout in [1, 0.75, 0.5]:
    #     print('Running check with dropout = ', dropout)
    #     model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
    #                               weight_scale=5e-2, dtype=np.float64,
    #                               dropout=dropout, seed=123)
    #     loss, grads = model.loss(X, y)
    #     print('Initial loss: ', loss)
    #
    #     for name in sorted(grads):
    #         f = lambda _:model.loss(X, y)[0]
    #         grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    #         print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))


    ##---------------------------Regularization experiment: pass!-----------------------------------------
    ##Train two identical nets, one with dropout and one without
    # np.random.seed(231)
    # num_train = 500
    # small_data = {
    #     'X_train': data['X_train'][:num_train],
    #     'y_train': data['Y_train'][:num_train],
    #     'X_val': data['X_val'],
    #     'y_val': data['Y_val'],
    # }
    #
    # solvers = {}
    # dropout_choices = [1, 0.25]
    # for dropout in dropout_choices:
    #     model = FullyConnectedNet([500], dropout=dropout)
    #     print(dropout)
    #
    #     solver = Solver(model, small_data,
    #                     num_epochs=25, batch_size=100,
    #                     update_rule='adam',
    #                     optim_config={
    #
    #                         'learning_rate': 5e-4,
    #                     }, verbose=True, print_every=100)
    #     solver.train()
    #     solvers[dropout] = solver
    #
    # train_accs = []
    # val_accs = []
    # for dropout in dropout_choices:
    #     solver = solvers[dropout]
    #     train_accs.append(solver.train_acc_history[-1])
    #     val_accs.append(solver.val_acc_history[-1])
    #
    # plt.subplot(2,1,1)
    # for dropout in dropout_choices:
    #     plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
    # plt.title('train accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(ncol=2, loc='lower right')
    #
    # plt.subplot(2,1,2)
    # for dropout in dropout_choices:
    #     plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
    # plt.title('val history')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend(ncol=2, loc='lower right')
    #
    # plt.gcf().set_size_inches(15, 15)
    # plt.show()

