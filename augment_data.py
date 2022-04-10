import numpy as np
import random
import scipy.io
from PIL import Image

def augment_data(X, y, ratio:float):
    def change(mat):
        def rotate(img):  # 在-30至30度中随机旋转
            deg = -30 + 60 * random.random()
            return img.rotate(deg)

        def translate(img):  # 随机放大放小与移动
            s0 = img.size[0]
            s = int(s0 * (random.random() * 0.5 + 0.5))
            res = Image.new(img.mode, img.size)
            res.paste(img.resize((s, s)), (int(random.random() * (s0 - s)), int(random.random() * (s0 - s))))
            return res

        img = Image.fromarray(mat)

        img = translate(rotate(img))

        return np.array(img)


    n_instances, n_vars = X.shape
    img_size = n_vars ** 0.5
    assert not img_size % 1 # 为整数
    img_size = int(img_size)

    augment = int(n_instances * ratio)
    index = [int(n_instances * random.random()) for i in range(augment)]

    x_add = np.array([change(X[i].reshape(img_size, img_size)).reshape(img_size**2) for i in index])
    y_add = np.array([y[i] for i in index])

    return np.concatenate((X, x_add), axis=0), np.concatenate((y, y_add), axis=0)


if __name__ == '__main__':
    data = scipy.io.loadmat('D:\Projects\Python\studyDeepLearning\hw1_simple_network\digits.mat') # Xvalid, yvalid, x_train, yExpanded, Xtest, ytest

    X = data['x_train']
    y = data['y_train']

    res = augment_data(X, y, 0.7)




