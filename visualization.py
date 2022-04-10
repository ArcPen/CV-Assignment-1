import numpy as np
import matplotlib.pyplot as plt
from math import ceil


def vis_loss_curve(loss_train, loss_test, max_iter=10000):
    x_train = np.linspace(0, max_iter, len(loss_train))
    x_test = np.linspace(0, max_iter, len(loss_test))

    fig, ax = plt.subplots()
    ax.plot(x_train, loss_train, label='train loss')
    ax.plot(x_test, loss_test, label='test loss')
    ax.set(ylim=(0, min((np.max(loss_test)*8), np.max(loss_train))))
    plt.title('Loss Value')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    pass

def vis_accuracy_curve(accuracy, max_iter=10000):
    x_axis = np.linspace(0, max_iter, len(accuracy))

    fig, ax = plt.subplots()
    ax.plot(x_axis, accuracy)
    ax.set(ylim=(0,1))
    plt.title("Accuracy on test set")
    plt.xlabel("Iteration")
    plt.ylabel("(1-Accuracy)")
    plt.show()


def vis_weights(model):
    from MLPclassification import form_weights
    from data_readin import x_test, y_test
    _, nVars = x_test.shape
    nLabels = len(np.unique(y_test))
    hiddenBias, hiddenWeights, inputBias, inputWeights, outputWeights = \
        form_weights(model.w, model.bias, model.nHidden, nLabels, nVars)

    # %%
    r = ceil(model.nHidden[0] ** 0.5)
    plt.figure()
    plt.suptitle('Weights of layer 1')
    for i in range(model.nHidden[0]):
        w_temp = inputWeights[:, i].reshape(28, 28).T
        plt.subplot(r, r, i + 1)
        plt.imshow(w_temp)
        plt.axis('off')
    # plt.tight_layout()
    plt.show()

    # %%
    w_combined = []
    for i in range(nLabels):
        w_combined.append(inputWeights @ outputWeights[:, i:i + 1])

    plt.figure()
    plt.suptitle("Weights of last layer")
    for i, mat in enumerate(w_combined):
        plt.subplot(2, 5, i + 1)
        plt.imshow(mat.reshape(28, 28).T)
        plt.title(f'Label {i}')
        plt.axis('off')
    plt.show()

#%% main
if __name__ == '__main__':
    # loss_train = np.array([6411.9, 53.125, 4.835, 3.846, 3.977, 2.101])
    # loss_test = np.array([16.947, 0.8955, 0.7725, 0.7459, 0.7222])
    # accuracy = [0.9127, 0.3298, 0.2316, 0.1953, 0.1743, 0.1639, 0.1582, 0.1566, 0.1503, 0.1453, 0.1423]
    # # vis_loss_curve(loss_train, loss_test)
    # vis_accuracy_curve(accuracy)
    from neuralNetworkModel import NeuralNetworkModel

    # model = NeuralNetworkModel.from_save_data('model_25_04101815.dat', in_save_folder=True)
    model = NeuralNetworkModel.from_save_data('model_46_04102003.dat', in_save_folder=True)

    vis_weights(model)

    # plt.imshow(x_test[11].reshape(28, 28).T)
    # plt.show()