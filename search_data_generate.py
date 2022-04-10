from neuralNetworkModel import NeuralNetworkModel
from random import random
from math import log

def generate_data():
    '''
    Generate data in the ./save_data folder. Randomly set these hyperparameters.
    :return:
    '''

    for i in range(20):
        init_learnrate = 10 ** (-5*random())
        hidden_layer = int(200*random()) + 100
        reg_param = random()*3

        print(f"{i:02d} ir: {init_learnrate:e}, hidden_layer {hidden_layer}, reg_param {reg_param:.3f}")
        model = NeuralNetworkModel()
        model.nHidden = [hidden_layer]
        model.init_learnrate = init_learnrate
        model.regularization = reg_param

        model.max_iter = 20000
        model.print_log = False
        model.use_tqdm = True

        model.train()
        model.info()

if __name__ == '__main__':
    generate_data()