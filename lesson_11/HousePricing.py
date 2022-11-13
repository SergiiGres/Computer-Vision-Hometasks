# Introduction to Neural Network
# Using Tensorflow Keras In this notebook we are going to implement our first neural
# network using Tensorflow (Keras). We will apply the network to the classic house pricing problem.

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

# For reproducibility reasons, we will set the seeds of the random number generators that are internally used
# throughout the notebook. This way we make sure that different runs of this notebook will always produce the same
# results. This is helpful if we want to evaluate the effect of a change that we have introduced, i.e., to compare
# the performance before and after the implementation of the change.

from numpy.random import seed
from tensorflow.random import set_seed

seed_value = 1234578790
seed(seed_value)
set_seed(seed_value)

# The Classic House Pricing Problem This classic task consists of estimating the price of a house (USA) based on the
# available description and attributes, i.e., the living area, number of bathrooms, construction year, etc. You can
# download the dataset from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data .
#
# The dataset is stored in a comma separated file (csv). Let's load the dataset (the train split) and briefly explore
# it.

dataset = pd.read_csv('/Users/sergiigres/PycharmProjects/ComputerVision/Computer-Vision-Hometasks/lesson_11/data'
                      '/train.csv')

# Let's print the loaded data. We'll see that each row corresponds to a sample (i.e., a particular house) and each
# column represents a certain attribute. There are 80 attributes in total (including the price). In this exercise, we
# will use a subset of them.

print(dataset)

# Let's now have a quick look on the house prices. The pandas module already provides us with a handy function to do
# this :-)

dataset['SalePrice'].describe()

# Training Our First Neural Network Data Preparation First, we are going to select the attributes we want to use for
# predicting the price. We will start very simple, with the living area (in square feet). Note that since not all
# samples (houses) have all of the attributes filled in, we will fill the missing values by the mean value of the
# dataset. If the dataset is complete, we don't have to do this.

features = ['SalePrice', 'GrLivArea']
data = dataset[features]

# Filling nan with the mean of the column
data = data.fillna(data.mean())

# Our inptus, i.e., the living areaa (in square feet) are usually in the order of 1e3. However, the outputs,
# i.e. the house prices, are usually in the order of 1e5. This is a huge mismatch in values. Even though the network
# will learn to cope with this, it is always a good idea to bring the inputs (all of them) and the outputs to the
# same order of magnitude, typically around 0 or 1. Therefore, we are going to normalize our data. We will learn more
# about data normalization in the next lesson.

# Extract input values (living area) and normalize
# The dimensionality of the input data must always be explicit
x = data[[features[1]]]
mu, sigma = np.mean(x), np.std(x)
x = (x - mu) / sigma

# Extract output values (prices) and normalize
y = data[features[0]].values
y = y / 100000

# Split into 75% for train and 25% for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Building the Network We are going to build our first neural network, a very simple one. For that, we will need to
# import the necessary modules.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import metrics

# Let's build the model using the Sequential API from Tensorflow.

model = Sequential()
model.add(Dense(1, input_dim=x.shape[1], activation='relu'))

# When a model is built, it creates a computation graph. This graph needs to be compiled before we can start any
# trainings. In the compilation, we will tell Tensorflow what optimizer to use, what is the loss function and what
# metrics shall be used for monitoring the training. We can also define the list of callbacks to have further control
# over the training procedure but this is out of the scope of this lecture.

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.mae])

# Finally , let's visualize the model.
model.summary()

# This is a simple model with only one neuron. As we know, this neuron has two parameter: weight and bias.

# Training the Network
# Let's now train the network we have just built. In Keras this is super easy, it is a one line command!

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=150, batch_size=32)


# For better visualization, we are going to implement a function to show the training history and printing the
# training error as well as the absolute price prediction error (on the test split).

def plot_history(history):
    h = history.history
    epochs = range(len(h['loss']))

    plt.subplot(121), plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    plt.subplot(122), plt.plot(epochs, np.array(h['mean_absolute_error']) * 1e5, '.-',
                               epochs, np.array(h['val_mean_absolute_error']) * 1e5, '.-')
    plt.grid(True), plt.xlabel('epochs'), plt.ylabel('MAE')
    plt.legend(['Train', 'Validation'])

    print('Train MAE     ', h['mean_absolute_error'][-1] * 1e5)
    print('Validation MAE', h['val_mean_absolute_error'][-1] * 1e5)


plot_history(history)

# We see that the model makes an estimation error of almost 40 000 USD. That's quite a lot but have in mind that we
# have only used the living area as the parameter and a model of only one neuron. In fact, what does this model do?
# Let's print what the model has learnt.
#
model.layers[0].get_weights()
# Let's now run the model to predict the value of a random house.

idx = 50
pred = model.predict(x_test.iloc[[idx]])
print(pred, y_test[idx])

# Let's Improve the Model Given that the model is extremely simple, the performance is not that bad. But we shall be
# able to do better. Therefore, let's try to use more attributes and a bigger network.
#
# First, we are going to use (as inputs) more attributes (or features) for price estimation. Let's consider,
# for instance, the following ones:
#
# OverallQual Rates the overall material and finish of the house GrLivArea Above grade (ground) living area square
# feet GarageCars Size of garage in car capacity FullBath Full bathrooms above grade YearBuilt Original construction
# date Data Preparation As we did before, we are going to prepare the data but this time with all 5 attributes (
# features). For simplicity, we are going to use the StandardScaler from sklearn. But don't worry, it does exactly
# the same thing as we did before, it's just easier to use for multi-dimensional data :-)

features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'FullBath', 'YearBuilt']
data = dataset[features]

# Filling nan with the mean of the column:
data = data.fillna(data.mean())

# Extract input values and normalize
x = data[features[1:]]
scale = StandardScaler()
x = scale.fit_transform(x)

# Extract output values (prices) and normalize
y = data[features[0]].values
y = y / 100000

# Split into 75% for train and 25% for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Building the Network We are going to use the same 1-layer network. But remember, since we have now 5 features as
# inputs, the network (neuron in this case) makes a linear combination of all of them and adds the bias. Therefore,
# we have now 6 learnable (trainable) parameters, i.e. 5 weights plus bias.

model = Sequential()
model.add(Dense(1, input_dim=x.shape[1], activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.mae])
model.summary()

# Training the Network
# We changed the data but the interface stays the same. Running the training is exactly the same as before!

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=150, batch_size=32, verbose=0)
plot_history(history)

# Let's also see what the network has learnt about the feature relevance.

print(features[1:])
print(model.layers[0].get_weights())

# Keras Functional API Besides the Sequential API that we have been using so far, Keras also offers a different way
# to define the models. The Functional API is in general more verstile and it allows for branching (which the
# Sequential does not). In order to use the Functional API, we need to make two more imports (note that we don't need
# to import the Sequential anymore).

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

# Deep Neural Network
# Let's now build our fisr deep neural network (DNN). Still a very simple one but consisting of 2 layers.

inputs = Input(shape=x.shape[1])
outputs = Dense(5, activation='relu')(inputs)
outputs = Dense(1, activation='linear')(outputs)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[metrics.mae])
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=150, batch_size=32, verbose=0)
plot_history(history)
