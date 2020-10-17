import os
# print(os.listdir("./input/MNIST"))
import tensorflow as tf
# libraries
import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O(e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()   # setting seaborn default for plots

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.datasets import mnist

# for Convolutional Neural Network (CNN) model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler

train = pd.read_csv('./input/MNIST/train.csv')
print(train.shape)
print(train.head())

test = pd.read_csv('./input/MNIST/test.csv')
print(test.shape)
print(test.head())