# from math import tanh, cosh
from math import e, exp, log
from random import random

from math import tanh, cosh
dtanh = lambda x: 1 - tanh(x) ** 2

# tanh = lambda x: (e**x - e**(-x)) / (e**x + e**(-x))
# cosh = lambda x: (e**x + e**(-x)) / 2
# dtanh= lambda x: 1 - (e**x - e**(-x))**2 / (e**x + e**(-x))**2

import numpy as np
# import cupy as np


def form_weights(w, bias, nHidden, nLabels, nVars):
    inputWeights = w[:nVars * nHidden[0]].reshape(nVars, nHidden[0])  # 输入层权重
    inputBias = bias[:nHidden[0]]
    offset = nVars * nHidden[0]
    offset_bias = nHidden[0]
    hiddenWeights = []
    hiddenBias = []
    for h in range(1, len(nHidden)):
        hiddenWeights.append(
            w[offset:offset + nHidden[h - 1] * nHidden[h]].reshape(nHidden[h - 1], nHidden[h])
        )
        offset = offset + nHidden[h - 1] * nHidden[h]
        hiddenBias.append(
            bias[offset_bias:offset_bias + nHidden[h]]
        )
        offset_bias += nHidden[h]
    outputWeights = w[offset:offset + nHidden[-1] * nLabels]
    outputWeights = outputWeights.reshape(nHidden[-1], nLabels)
    return hiddenBias, hiddenWeights, inputBias, inputWeights, outputWeights


def MLPclassificationLoss(
        w:np.ndarray, bias:np.ndarray, X:np.ndarray, y:np.ndarray, nHidden:list, nLabels:int,
        regularization=0, dropout=0, nargout:int=2
    ):
    '''
    :param w:
    :param bias: 偏置参数
    :param X:
    :param y:
    :param nHidden: 隐藏层参数
    :param nLabels:
    :param regularization: 正则化参数
    :param dropout: 每一层dropout的概率, 可以取0.5
    :param nargout: 输出的参数数量。1为只输出loss，2为loss + gradient
    :return:
    '''
    nInstances, nVars = X.shape

    # Form Weights
    hiddenBias, hiddenWeights, inputBias, inputWeights, outputWeights = \
        form_weights(w, bias, nHidden, nLabels, nVars)

    f = 0
    if nargout > 1:
        gInput = np.zeros(inputWeights.shape)
        gHidden = []
        for h in range(1, len(nHidden)):
            gHidden.append(np.zeros(hiddenWeights[h-1].shape))
        gOutput = np.zeros(outputWeights.shape)

    # Compute Output
    f_tanh = np.vectorize(tanh)
    f_sech_sqr = np.vectorize(dtanh)  # derivative of tanh(x)
    for i in range(1, nInstances+1):
        ip = [X[i-1:i,:] @ inputWeights + inputBias,]
        if dropout: ip[-1] *= np.random.rand(*ip[-1].shape) < (1-dropout)
        fp = [f_tanh(ip[0]),]
        for h in range(1, len(nHidden)):
            ip.append(fp[h-1] @ hiddenWeights[h-1] + hiddenBias[h-1])
            if dropout: ip[-1] *= np.random.rand(*ip[-1].shape) < (1 - dropout)
            fp.append(np.vectorize(tanh)(ip[h]))
        yhat = fp[-1] @ outputWeights

        # softmax
        yexp = np.vectorize(exp)(yhat)
        yprob = yexp / np.sum(yexp)

        # relativeErr = yhat - yExpanded[i-1:i, :]
        # f = f + (relativeErr @ relativeErr.T)[0,0]
        f = f + -log(yprob[(y[i-1]==1).reshape(1, -1)][0])

        if not nargout>1:
            continue
        # err = 2 * relativeErr
        dloss = yprob + (y[i-1]==1).reshape(1, -1) * -1

        # Output Weights
        # for c in range(nLabels):
        #     gOutput[:,c] = gOutput[:,c] + dloss[0,c] * np.transpose(fp[-1][0])
        gOutput += fp[-1].T @ dloss

        if len(nHidden) > 1:
            # Last Layer of Hidden Weights
            # backprop = np.zeros((nLabels, nHidden[-1]))
            # for c in range(nLabels):
            #     backprop[c, :] = dloss[0, c] * np.multiply(f_sech_sqr(ip[-1]), outputWeights[:, c].transpose())
            #     gHidden[-1] = gHidden[-1] + fp[-2].transpose() @ backprop[c:c+1, :]
            # backprop = np.sum(backprop, axis=0)
            backprop = (f_sech_sqr(ip[-1].T) * outputWeights @ dloss.T).T
            gHidden[-1] = fp[-2].T @ backprop

            # Other Hidden Layers
            for h in range(len(nHidden)-2, 0, -1):
                backprop = np.multiply(backprop @ hiddenWeights[h].T, f_sech_sqr(ip[h]))
                gHidden[h-1] = gHidden[h-1] + fp[h-1].T @ backprop

            # Input Weights
            backprop = np.multiply(backprop @ hiddenWeights[0].T, f_sech_sqr(ip[0]))
            gInput = gInput + X[i-1:i, :].T @ backprop
        else:
            # Input Weights
            # for c in range(nLabels):
            #     gInput = gInput + dloss[0,c] * x_train[i-1:i, :].T @ np.multiply(f_sech_sqr(ip[-1]), outputWeights[:,c].T)
            gInput += X[i-1:i, :].T @ (f_sech_sqr(ip[-1].T) * outputWeights @ dloss.T).T

    # Put Gradient into vector
    if nargout > 1:
        g = np.zeros(w.shape)
        g[:nVars*nHidden[0]] = gInput[:].reshape(1, nVars*nHidden[0])
        offset = nVars * nHidden[0]
        for h in range(1, len(nHidden)):
            g[offset:offset + nHidden[h-1]*nHidden[h]] = gHidden[h-1].reshape(1, nHidden[h-1]*nHidden[h])
            offset = offset + nHidden[h-1]*nHidden[h]
        g[offset:offset + nHidden[-1]*nLabels] = gOutput[:].reshape(1, nHidden[-1]*nLabels)

    # Regularization
    if regularization:
        f = f + regularization * np.sum(np.square(w))
        if nargout == 2:
            g = g + regularization * w

    if nargout == 1:
        return f
    elif nargout == 2:
        return f, g
    else:
        raise ValueError

def MLPFineTuning(w, bias, X, yExpanded, nHidden, nLabels):
    nInstances, nVars = X.shape

    # Form Weights
    hiddenBias, hiddenWeights, inputBias, inputWeights, outputWeights = \
        form_weights(w, bias, nHidden, nLabels, nVars)

    # Compute input to the last layer
    inp = np.zeros((nInstances, nHidden[-1]))
    for i in range(1, nInstances):
        ip = [X[i-1:i, :] @ inputWeights + inputBias, ]  # np.matrix[]
        fp = [np.vectorize(tanh)(ip[0]), ]  # np.matrix[]
        for h in range(1, len(nHidden)):
            ip.append(fp[h - 1] @ hiddenWeights[h - 1] + hiddenBias[h-1])
            fp.append(np.vectorize(tanh)(ip[h]))
        # yhat[i-1:i, :] = fp[-1] @ outputWeights
        inp[i-1:i, :] = fp[-1] # fp[-1] is the input

    xi_xit = sum(inp[i-1:i, :].T @ inp[i-1:i, :] for i in range(1, nInstances))
    xi_yit = sum(inp[i-1:i, :].T @ yExpanded[i - 1:i, :] for i in range(1, nInstances))
    outputWeights = np.linalg.inv(xi_xit) @ xi_yit

    print("Fine tuning of last layer complete.")
    w[-(nHidden[-1] * nLabels):] = outputWeights.reshape(nHidden[-1]*nLabels)
    return w

def MLPclassificationPredict(w:np.ndarray, bias:np.ndarray, X:np.ndarray, nHidden:list, nLabels:int):
    nInstances, nVars = X.shape

    # Form Weights
    hiddenBias, hiddenWeights, inputBias, inputWeights, outputWeights = \
        form_weights(w, bias, nHidden, nLabels, nVars)

    # Compute Output
    yhat = np.zeros((nInstances, nLabels))
    for i in range(1, nInstances):
        ip = [X[i-1:i, :] @ inputWeights + inputBias, ]  # np.matrix[]
        fp = [np.vectorize(tanh)(ip[0]), ]  # np.matrix[]
        for h in range(1, len(nHidden)):
            ip.append(fp[h - 1] @ hiddenWeights[h - 1] + hiddenBias[h-1])
            fp.append(np.vectorize(tanh)(ip[h]))
        yhat[i-1:i, :] = fp[-1] @ outputWeights

    y = np.argmax(yhat, axis=1)
    return y.reshape(nInstances, 1)

