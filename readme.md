# DATADATA130051.01 CV HW1

This is the project of the first homework of Computer Vision course.

## Training Process

Run `search_data_generate.py` to do the stochastic search of hyper parameters. You can also modify the search space and search parameters. Data would be generated in the 'save_data' folder. 

## Find the best model

Run `search_data_analysis.py`, and it will show you about the good models.

The best model is also put here in the repo. [Link](./model_best_one.dat)

## Test Process

If you would like to use the model, first clone the repo, then load the model using this code:

```python
from neuralNetworkModel import NeuralNetworkModel
model = NeuralNetworkModel.from_save_data(filepath_to_save_data)
```

Then, you can happily predict any data you want. 

```python
# input should suit the shape of (n, 784)
# if you have images, you would want to reshape them to 1-d vectors
model.predict(x_input)
```





