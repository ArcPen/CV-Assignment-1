import pickle

from neuralNetworkModel import NeuralNetworkModel
from random import random
from math import log
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    show_hist = True

    list_filename = os.listdir('save_data')

    list_accuracy = []
    for filename in list_filename[0:]:
        with open(f"./save_data/{filename}", 'rb') as fp:
            data = pickle.load(fp)
            list_accuracy.append(1-data['accuracy'])

    if show_hist:
        plt.hist(list_accuracy, bins=40, range=(0.7, 1), edgecolor='black')
        plt.title("accuracy distribution of trained models".title())
        plt.xlabel("Accuracy")
        plt.ylabel("Counts")
        plt.show()

    good_models = [list_filename[i] for i in range(len(list_filename)) if list_accuracy[i] > 0.9]
    model_data = []
    for filename in good_models:
        with open(f"./save_data/{filename}", 'rb') as fp:
            model_data.append(pickle.load(fp))

    model_data.sort(key=lambda d: d['accuracy'])

    for d in model_data:
        print(f"acc: {1 - d['accuracy']:.4f} lr: {d['init_learnrate']:e}"
              f" reg: {d['regularization']:.5f} nHid: {d['nHidden'][0]:3d} time: {d['time_elapsed'][2:7]}")

    best_model = NeuralNetworkModel.from_save_data_dict(model_data[0])
    best_model.visualize()

    for filename in good_models:
        with open(f"./save_data/{filename}", 'rb') as fp:
            d = pickle.load(fp)
            if d['accuracy'] == best_model.accuracy:
                print('Best model is', filename)