# -*- coding: utf-8 -*-
"""crispr_2023_crispr_exercise3

This file is part of the CRISPRsummerschool 2023 exercises
Copyright (c) 2023 Christian Anthon

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, version 3.

# Convolutions in CRISPR on-target
Below you will find the third exercise.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2023/CRISPR/exercise/crispr_2023_crispr_exercise3.ipynb)

In this exercise we will take a look at what are the actual outcome of the convolutions of the on-target sequence in the deep learning model.

In the code below, we will re-use the simplified version of the CRISPRon ontarget model we created in the previous exercise, however we have reduced the number of convolutions to 10.

## basic code definitions
Enter the cell below and press play or Ctrl+Enter in the block below to execute. You should see the message "Definitions executed" printed after execution.
"""

#!/usr/bin/env python3
from google.colab import drive
from random import randint
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pickle
import urllib3
import subprocess
import os

from tensorflow.keras import models, callbacks, Model, Input, utils, metrics
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, concatenate

eLENGTH30 = 30
eDEPTH = 4
# Function to onehot encode the data
def onehot(x):
    z = list()
    for y in list(x):
        if y in "Aa":
            z.append(0)
        elif y in "Cc":
            z.append(1)
        elif y in "Gg":
            z.append(2)
        elif y in "TtUu":
            z.append(3)
        else:
            print("Non-ATGCU character in", x)
            raise Exception
    return z

# Function to set the data into the appropriate format
def set_data(DX, s, mask=None):
  #mask should be a list length len(s) consisting of 1s and 0s, only positions where mask is 0 will be onehot encoded, those with 1s will be masked out.
    if s is None:
        return
    assert(mask==None or (type(mask) is list and len(mask) == len(s)))
    if type(mask) is list:
        for j, x in enumerate(onehot(s)):
            if mask[j] == 0:
                DX[j][x] = 1
    else:
        for j, x in enumerate(onehot(s)):
            DX[j][x] = 1


# Preprocessing function for the sequence data
def preprocess_seq(data, mask=None, use_dgb=True):
    DATA_X30 = np.zeros((len(data), eLENGTH30, eDEPTH), dtype=np.float32)  # onehot
    DATA_G = np.zeros((len(data), 1), dtype=np.float32)  # deltaGb
    DATA_Y = np.zeros((len(data)), dtype=np.float32)  # efficiency

    for l, d in enumerate(data):
        set_data(DATA_X30[l], d[0], mask)
        if use_dgb:
            DATA_G[l] = -d[1]
        DATA_Y[l] = d[2]
    return (DATA_X30, DATA_G, DATA_Y)

print("Definitions executed")

#commands run to download data
#You may need to change the URL to a suitable one outside of github due to ratelimits
! curl -o training_data.pickle https://github.com/RTH-tools/CRISPRsummerschool/raw/main/2023/CRISPR/exercise/training_data.pickle
! curl -o validation_data.pickle https://github.com/RTH-tools/CRISPRsummerschool/raw/main/2023/CRISPR/exercise/validation_data.pickle

# Training Data
PATH = './'
with open(PATH+'/training_data.pickle', 'rb') as f:
    d = pickle.load(f)

#Validation data read
with open(PATH+'/validation_data.pickle', 'rb') as f:
    dv = pickle.load(f)

print('Data loaded')
OPT = 'adam' #use the ADAM optizer
LOSS = 'mse' #loss function is mean squared error

DROPOUT_DENSE = 0.3


CONV_1_SIZE = 3
N_CONV_1 = 10
N_DENSE = 40
N_OUT = 40

# Inputs
inputs_30 = list()

inputs_c_30 = Input(shape=(eLENGTH30, eDEPTH), name="inputs_onehot_30")
inputs_30.append(inputs_c_30)
inputs_g = Input(shape=(1), name="inputs_dgb_on")
inputs_30.append(inputs_g)

# Model_30 layers
for_dense_30 = list()

# First convolution layer
conv1_out_30 = Conv1D(N_CONV_1, CONV_1_SIZE, activation='relu', input_shape=(eLENGTH30, eDEPTH), name="conv_3_30")(inputs_c_30)
conv1_flatten_out_30 = Flatten(name="flatten_3_30")(conv1_out_30)
for_dense_30.append(conv1_flatten_out_30)

# Concatenation of conv layers and deltaGb layer
concat_out_30 = concatenate(for_dense_30, name="concat_cnv_30")

# First dense (fully connected) layer
dense0_out_30 = Dense(N_DENSE, activation='relu', name="dense_0_30")(concat_out_30)
dense0_dropout_out_30 = Dropout(DROPOUT_DENSE, name="drop_d0_30")(dense0_out_30)

# Gb input used raw
concat1_out_30 = concatenate((inputs_g, dense0_dropout_out_30), name="concat_dgb_30")

# First dense (fully connected) layer
dense1_out_30 = Dense(N_DENSE, activation='relu', name="dense_1_30")(concat1_out_30)
dense1_dropout_out_30 = Dropout(DROPOUT_DENSE, name="drop_d1_30")(dense1_out_30)

# Second dense (fully connected) layer
dense2_out_30 = Dense(N_OUT, activation='relu', name="dense_2_30")(dense1_dropout_out_30)
dense2_dropout_out_30 = Dropout(DROPOUT_DENSE, name="drop_d2_30")(dense2_out_30)

# Output layer
dense_on_off_out = Dense(N_OUT, activation='relu', name="dense_on_off")(dense2_dropout_out_30)
dense_on_off_dropout_out = Dropout(DROPOUT_DENSE, name="drop_d_on_off")(dense_on_off_out)
output_30 = Dense(1, name="output_30")(dense_on_off_dropout_out)

# Model_30
postfix = '.model_30'
inputs = inputs_30
outputs = [output_30]
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss=LOSS, optimizer=OPT, metrics=['mae', 'mse'])
model.summary()

print('Model defined')


#training data preprocess
(x30, g, y) = preprocess_seq(d)
#Validation data preprocess
(x30v, gv, yv) = preprocess_seq(dv)

"""## Exercise 3.1 Model definition and convolutions
Look at the model summary in the output above.

What is the dimensions (shape) of the output of the convolutional layer?

How does the output shape relate to the input shape, the size of the convolutions, and to the number of convolutions? (*Hint patching*)

"""

#answer

"""## Exercise 3.2 Model training

Execute the model **training** to initialize weights for later use. Check that you get more or less the same result as in exercise 1.
"""

print("training...")

OPT = 'adam' #use the ADAM optizer
LEARN = 1e-4 #learning rate
EPOCHS = 200 #maximum number of Epochs
LOSS = 'mse' #loss function is mean squared error
BATCH_SIZE = 64 #batch size for the training


#early stopping is a way to control for overfitting and save the best model
es = callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.1,
            patience=25,
            verbose=1,
            mode='auto',
            restore_best_weights=True,
            ),

#save the initial weights to be able restart the training using the same weights each time
model.save_weights('model_30.init.h5')

# Training
history = model.fit(
    (x30, g), y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=((x30v,gv), yv),
    callbacks=[es],
    verbose=1
)


print("done")

print("evaluating validation data using best model weights")
model.evaluate((x30v,gv), yv)

"""##Exercise 3.3
Use ``` tf.keras.Model.get_layer``` to get the convolutional layer above by name and ``` tf.keras.layers.Layer.get_weights```to get the convolutional kernel weights of the untrained model?

The weights are af list of two arrays. The first array is the actual weights, while the second array is the biases, which we can ignore for now. Print the first array from the weights you have just obtained?

What is the dimension (shape) of the first of the 10 convolutions? (*Hint you may find it easier to look at the weights after the array has been transposed*)
"""

#answer

"""## Exercise 3.4
Below we transform the ontarget sequence for the first guide and transform it into a form suitable as input the convolutional layer, which expect an extra dimension to accomodate a list of inputs. That is shape(30,4) for the first input is added a dimension to become shape(1,30,4)
"""

print(x30[0].shape)
x30_0 = x30[0].copy()
x30_0 = x30_0.reshape([1, 30, 4])
print(x30_0.shape)

"""###Exercise 3.4.1
Apply the convolutional layer on the first ontarget sequence (```x30_0```) and examine the result.

What is the output dimensions (shape)?

Identify the array which is the resulting from applying the first of the 10 convolutions?

In the output you will see only positive numbers and a lot of zeroes. Why is that, when the convolutions may have both positive and negative weights?
"""

#answer

"""Below we define a very simple convolutional layer with just one convolution of size 3 without biases"""

#array defining the weights and biases (not used)
w = [np.array([
    [[ 0.5], [ 0.0], [ 0.0], [ 0.0]],
    [[ 0.5], [ 0.0], [ 0.0], [ 0.0]],
    [[ 0.0], [ 0.0], [ 0.0], [ 0.0]]])]
print(w[0].transpose())
#define a simple convolutional layer with just one convolution of size 3 with the w as the weights
conv_simple_1_3 = tf.keras.layers.Conv1D(1, 3, use_bias=False, weights=w, input_shape=(30,4))

"""### Exercise 3.4.2
Apply the simple convolution defined above on the first ontarget ```(x30_0)``` and relate the result to the input sequence of the first ontarget?

Convolutions are sometimes called filters. How does your observation of the result of the convolution match up with that?
"""

#answer

"""##Exercise 3.5
Convolutions are just one way of finding patterns and features of the inputs in deep learning. One of the latest / greatest invention in the area of deep learning is the natural language processing models like e.g. ```BERT```.

What is the philosophy behind natural language processing?

Do you think that fx natural language processing or LSTM models are suitable for CRISPR ontarget modeling

And what about for CRISPR offtarget and specificity modeling?


"""

#answer
