# -*- coding: utf-8 -*-
"""crispr_2023_crispr_exercise1

This file is part of the CRISPRsummerschool 2023 exercises
Copyright (c) 2023 Christian Anthon

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, version 3.

# Warming up exercise
Below you will find the first exercise, in which you will be introduced a small CRISPR on-target model in Tensorflow / Keras and use it to train a small ontarget efficiency model on real data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2023/CRISPR/exercise/crispr_2023_crispr_exercise1.ipynb)

## basic code definitions
Enter the cell below and press play or Ctrl+Enter in the block below to execute. You should see the message "Definitions executed" printed after execution. (<5 minutes)
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

# Function to convert DNA sequence to one-hot encoding
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
def set_data(DX, s):
    if s is None:
        return
    for j, x in enumerate(onehot(s)):
        DX[j][x] = 1


# Preprocessing function for the sequence data
def preprocess_seq(data):
    DATA_X30 = np.zeros((len(data), eLENGTH30, eDEPTH), dtype=np.float32)  # onehot
    DATA_G = np.zeros((len(data), 1), dtype=np.float32)  # deltaGb
    DATA_Y = np.zeros((len(data)), dtype=np.float32)  # efficiency

    for l, d in enumerate(data):
        set_data(DATA_X30[l], d[0])
        DATA_G[l] = -d[1]
        DATA_Y[l] = d[2]
    return (DATA_X30, DATA_G, DATA_Y)
#commands run to download data
#You may need to change the URL to a suitable one outside of github due to ratelimits
! curl -o training_data.pickle https://github.com/RTH-tools/CRISPRsummerschool/raw/main/2023/CRISPR/exercise/training_data.pickle
! curl -o validation_data.pickle https://github.com/RTH-tools/CRISPRsummerschool/raw/main/2023/CRISPR/exercise/validation_data.pickle
print('\n\nDefinitions executed')

"""## Exercise 1.1 (<5minutes)
The sequence of the ontarget of an example gRNA (ACTGAAAAAACCCCCTTTTT), needs to be onehot encoded. An example ontarget of ACTGAAAAAACCCCCTTTTT is TTTTACTGAAAAAACCCCCTTTTTGGGAAA, which includes a four nucleotide prefix, the ontarget to the gRNA, the PAM sequnce and a four nucleotide suffix.

Mark the prefix(4nt), on-target(20nt), PAM(3nt), and suffix (4nt) in the ontarget?

Use the onehot function defined above to get the encoding of ACTGAAAAAACCCCCTTTTT?

Is this what a onehot encoding is supposed to look like, and what could be wrong with encoding the sequence this way?
"""

#answer

"""
## Exercise 1.2 (<5 minutes)
Excecute the code below to load the data into the notebook."""

# x30 - onehot encoded 30mer
# g - deltaGb
# y - the efficiency value [0-100] )

# Training Data
PATH = './'
with open(PATH+'/training_data.pickle', 'rb') as f:
    d = pickle.load(f)
(x30, g, y) = preprocess_seq(d)

#Validation data read
with open(PATH+'/validation_data.pickle', 'rb') as f:
    dv = pickle.load(f)
(x30v, gv, yv) = preprocess_seq(dv)

"""### Exercise 1.2.1  (<5 minutes)
Get the first value of the raw unprocessed training data (d[0]) and the first values of the processed data (x30, g, y). Is this what you expected for the onehot encoding?
"""

#answer

"""### Exercise 1.2.2 Model definition (5-10 minutes)
In the code below, a simplified version of the CRISPRon ontarget model is defined. Review the code without diving into the details. Then execute it to load the model.
"""

OPT = 'adam' #use the ADAM optizer
LOSS = 'mse' #loss function is mean squared error

DROPOUT_DENSE = 0.3


CONV_1_SIZE = 3
N_CONV_1 = 40
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
model.is_trained = False

"""### Exercise 1.2.3   (<10 minutes)
Use model.summary and keras.utils.plot_model to review the model details. Where are inputs and outputs. Identify the convolutional and multi-layer perceptron parts of the model. Wheres is the ΔGb input inserted into the model?
"""

#answer

"""### Exercise 1.2.4 Model training (10 minutes)
Below you will find code for training the simplified CRISPRon model on the provided training data, using the validation data for model evaluation during training. Familiarize yourself with the code and parameters.

What is the difference between BATCH_SIZE and epochs?

Execute the model **training**
"""

print("training...")

OPT = 'adam' #use the ADAM optizer
LEARN = 1e-4 #learning rate
EPOCHS = 200 #maximum number of Epochs
LOSS = 'mse' #loss function is mean squared error
BATCH_SIZE = 64 #batch size for the training

# Training
history = model.fit(
    (x30, g), y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=((x30v,gv), yv),
    callbacks=[
        #early stopping is a way to control for overfitting and save the best model
        callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.1,
            patience=25,
            verbose=1,
            mode='auto',
            restore_best_weights=True,
            ),

    ],
    verbose=1
)
try:
  assert(model.is_trained == False)
except:
  print("Warning: The model was already trained, you need to re-execute the model definition in Exercise 1.2.2 to restart the training")

model.is_trained = True

print("done")
model.evaluate((x30v, gv), yv )

"""### Exercise 1.2.4.1 (15 minutes)
How many epochs did the code use before it stopped?

When did the training reach the optimimal model?

Does the code output the exact same performance if you run it twice? Why / Why not?
"""

#answer

"""### Exercise 1.2.4.2
Repeat the model initialization in (1.2.2) and the model training (1.2.4) 3-5 times and record the performance on the validation data each time.

For the best model you obtain, compare the mean squared error and mean absolute error on the validation data with the errors obtained for the **full model** trained on the same data (**mae 9.1, mse 141.3** for the **full model** which is based on the same principles, but contains more and larger layers)?
"""

# answer

"""### Exercise 1.2.5 (if time allows)
Play with the model and parameters to get a better performance.

Can you beat the full the model performance?

What would be the proper way to **test** that?
"""

#answer
