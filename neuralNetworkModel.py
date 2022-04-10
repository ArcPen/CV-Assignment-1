import pickle
from random import random
from math import cos, pi
from tqdm import tqdm
from datetime import datetime

import numpy as np

from standardizeCols import standardizeCols
from linearInd2Binary import linearInd2Binary
from augment_data import augment_data
from MLPclassification import MLPclassificationLoss, MLPclassificationPredict, MLPFineTuning


class NeuralNetworkModel:
    def __init__(self):
        # Model related configurations
        self.data_augmentation_ratio = 0.1
        self.w = None
        self.bias = None
        self.max_iter = 100000
        self.warmup_iter = 5000
        self.init_learnrate = 1e-4
        self.nHidden = [80,]
        self.regularization = 1.0
        self.dropout = 0.5
        self.momentum = 0.8
        self.batch_size = 10
        self.hist_loss_train = None
        self.hist_loss_test = None
        self.hist_accu_test = None

        # Miscellaneous configurations
        self.print_log = True
        self.use_tqdm = False
        self.use_data_augmentation = True

        # Some temp variables
        self.nLabels = None
        self.accuracy = None
        self.time_elapsed = None

    def train(self):
        time_start = datetime.now()

        from data_readin import x_train, y_train, x_test, y_test
        x_valid = x_test
        y_valid = y_test

        data_augmentation_ratio = self.data_augmentation_ratio
        # do data augmentation
        if data_augmentation_ratio:
            x_train, y_train = augment_data(x_train, y_train, ratio=data_augmentation_ratio)

        # overfitting test
        # x_train = x_test
        # y_train = y_test

        n, d = x_train.shape
        nLabels = len(np.unique(y_train))
        self.nLabels = nLabels
        yExpanded = linearInd2Binary(y_train, nLabels)
        y_test_expanded = linearInd2Binary(y_test, nLabels)
        t = x_valid.shape[0]
        t2 = x_test.shape[0]

        # Standardize columns and add bias
        (x_train, mu, sigma) = standardizeCols(x_train)
        x_train = np.insert(x_train, 0, np.ones(n), axis=1)
        d = d + 1

        # Make sure to apply the same transformation to the validation/test data
        x_valid = standardizeCols(x_valid, mu, sigma)[0]
        x_valid = np.insert(x_valid, 0, np.ones(t), axis=1)
        x_test = standardizeCols(x_test, mu, sigma)[0]
        x_test = np.insert(x_test, 0, np.ones(t2), axis=1)


        # Choose network structure
        nHidden = self.nHidden

        # Count number of parameters and initialize weights 'w'
        nParams = d * nHidden[0]
        for h in range(1, len(nHidden)):
            nParams = nParams + nHidden[h-1] * nHidden[h]
        nParams = nParams + nHidden[-1] * nLabels
        w = np.random.randn(nParams) if self.w is None else self.w
        bias = np.random.randn(sum(nHidden)) if self.bias is None else self.bias


        # Train with stochastic gradient descent
        print_log = self.print_log
        max_iter = self.max_iter
        warmup_iter = self.warmup_iter
        init_learnrate = self.init_learnrate
        learnrate_func = lambda iter: (
            init_learnrate if (iter < warmup_iter) else
            init_learnrate * (1 + cos(pi * (iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        )
        momentum = self.momentum
        regularization = self.regularization
        batch_size = self.batch_size
        dropout = self.dropout

        hist_loss_train = []
        hist_loss_test = []
        hist_accu_test = []
        funObj = lambda x,i: MLPclassificationLoss(
            w, bias, x_train[i:i + batch_size, :], yExpanded[i:i + batch_size, :], nHidden, nLabels,
            regularization=regularization, dropout=dropout, nargout=2
        )
        for iter in (tqdm(range(max_iter)) if self.use_tqdm else range(max_iter)):
            learnrate = learnrate_func(iter)

            # test the accuracy on the test set
            if (iter) % (max_iter // 20) == 0:
                yhat = MLPclassificationPredict(w, bias, x_valid, nHidden, nLabels)
                accuracy = np.sum(yhat != y_valid) / t
                hist_accu_test.append(accuracy)
                print(f"Iteration = {iter}, Error rate = "
                      f"{str(accuracy).ljust(6, '0')}", end='\t') if print_log else ''
                loss = MLPclassificationLoss(
                    w, bias, x_test, y_test_expanded, nHidden, nLabels,
                    regularization=regularization, dropout=dropout, nargout=1
                )
                loss = loss / x_test.shape[0]
                hist_loss_test.append(loss)
                print(f"Test avg loss: {loss:.5f}", end='\t') if print_log else ''


            i = int(random() * n)
            f, g = funObj(w, i)
            f = f / batch_size
            hist_loss_train.append(f)
            if print_log and (iter) % (max_iter // 20) == 0:
                print(f"Train avg loss: {f:.5f}") if print_log else ''
            if iter == 0:
                w_old = w
                w = w - learnrate * g
            else:
                w_new = w - learnrate * g + momentum * (w - w_old)
                w_old = w
                w = w_new

        w = MLPFineTuning(w, bias, x_train, yExpanded, nHidden, nLabels)
        yhat = MLPclassificationPredict(w, bias, x_valid, nHidden, nLabels)
        print(f"After fine tuning, validation error = "
              f"{str(np.sum(yhat != y_valid) / t).ljust(6, '0')}") if print_log else ''

        yhat = MLPclassificationPredict(w, bias, x_test, nHidden, nLabels)
        accuracy = np.sum(yhat != y_test) / t2
        hist_accu_test.append(accuracy)
        print(f"Test error with final model = {accuracy}")

        self.hist_loss_train = hist_loss_train
        self.hist_loss_test = hist_loss_test
        self.hist_accu_test = hist_accu_test

        # Save the model and data
        time_end = datetime.now()
        self.time_elapsed = str(time_end - time_start)
        self.accuracy = accuracy

        filename = "model_{}_{:02d}.dat".format(datetime.now().strftime("%m%d%H%M"), int(random()*100))
        with open(f'save_data/{filename}', 'wb+') as fp:
            data = dict(
                w=w,
                bias=bias,
                init_learnrate=init_learnrate,
                nHidden=nHidden,
                regularization=regularization,
                max_iter=max_iter,
                warmup_iter=warmup_iter,
                dropout=dropout,
                momentum=momentum,
                batch_size=batch_size,
                data_augmentation_ratio=data_augmentation_ratio,
                hist_loss_train=hist_loss_train,
                hist_loss_test=hist_loss_test,
                hist_accu_test=hist_accu_test,
                time_elapsed=str(time_end - time_start),
                accuracy=accuracy,
            )
            pickle.dump(data, fp)
        print("Data successfully saved to {}".format(filename))

    def predict(self, x, nLabels):
        yhat = MLPclassificationPredict(self.w, self.bias, x, self.nHidden, nLabels)
        return yhat

    def visualize(self):
        # Visualization part
        if (self.hist_loss_train and self.hist_loss_test and self.hist_accu_test) is None:
            raise Exception("Please run the model first to get the data, then visualize")
        from visualization import vis_loss_curve, vis_accuracy_curve, vis_weights

        vis_loss_curve(self.hist_loss_train, self.hist_loss_test, max_iter=self.max_iter)
        vis_accuracy_curve(self.hist_accu_test, max_iter=self.max_iter)
        vis_weights(self)

    def info(self):
        print(f"Training time {self.time_elapsed}, final error {self.accuracy}")

    def from_save_data(filename:str, in_save_folder=False):
        '''
        Create the model from saved data.
        '''
        model = NeuralNetworkModel()
        full_name = f'save_data/{filename}' if in_save_folder else filename
        with open(full_name, 'rb') as fp:
            data = pickle.load(fp)
        model.w = data['w']
        model.bias = data['bias']
        model.init_learnrate = data['init_learnrate']
        model.nHidden = data['nHidden']
        model.regularization = data['regularization']
        model.max_iter = data['max_iter']
        model.warmup_iter = data['warmup_iter']
        model.dropout = data['dropout']
        model.momentum = data['momentum']
        model.batch_size = data['batch_size']
        model.data_augmentation_ratio = data['data_augmentation_ratio']
        model.hist_loss_train = data['hist_loss_train']
        model.hist_loss_test = data['hist_loss_test']
        model.hist_accu_test = data['hist_accu_test']
        model.accuracy = data['accuracy']
        model.time_elapsed = data['time_elapsed']

        print("Model successfully loaded.")

        return model

    def from_save_data_dict(data:dict):
        '''
        Create the model from a dict that is loaded from save data.
        '''
        model = NeuralNetworkModel()

        model.w = data['w']
        model.bias = data['bias']
        model.init_learnrate = data['init_learnrate']
        model.nHidden = data['nHidden']
        model.regularization = data['regularization']
        model.max_iter = data['max_iter']
        model.warmup_iter = data['warmup_iter']
        model.dropout = data['dropout']
        model.momentum = data['momentum']
        model.batch_size = data['batch_size']
        model.data_augmentation_ratio = data['data_augmentation_ratio']
        model.hist_loss_train = data['hist_loss_train']
        model.hist_loss_test = data['hist_loss_test']
        model.hist_accu_test = data['hist_accu_test']
        model.accuracy = data['accuracy']
        model.time_elapsed = data['time_elapsed']

        print("Model successfully loaded.")

        return model




if __name__ == '__main__':
    model = NeuralNetworkModel()
    model.print_log = False
    model.use_tqdm = True
    model.max_iter = 20000
    model.train()
    model.visualize()
    model.info()


