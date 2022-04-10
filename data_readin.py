import scipy.io
import numpy as np

data_test = scipy.io.loadmat("MINIST/Test.mat")['Test']
data_train = scipy.io.loadmat("MINIST/Train.mat")['Train']

x_test = np.concatenate([i[0].reshape(1, -1) for i in data_test], axis=0)
y_test = np.concatenate([i[1] for i in data_test])

x_train = np.concatenate([i[0].reshape(1, -1) for i in data_train], axis=0)
y_train = np.concatenate([i[1] for i in data_train])

print("MINIST data loaded.")
