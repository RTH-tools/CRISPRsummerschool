# -*- coding: utf-8 -*-
"""crispr_2023_crispr_exercise2

This file is part of the CRISPRsummerschool 2023 exercises
Copyright (c) 2023 Christian Anthon

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, version 3.

# Extracting features from the deep learning results
Below you will find the second exercise.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2023/CRISPR/exercise/crispr_2023_crispr_exercise2.ipynb)

Deep learning does not easily lend itself to extraction of feature importance, like in the example of CRISPR where one could wish to know the importance of *e.g.* the first nucleotide of the NGG pam for the efficiency of the guide. In this exercise we will look at a way around this problem by masking out parts of the input sequence or of the energy parameter from the model input.

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

"""##Model definition and baseline performance (5-10 minutes)

---


In the code below, we will re-use the simplified version of the CRISPRon ontarget model we created in the previous exercise, but we will preprocess the data in different ways, by masking out information from the input in order to glean its relative importance.
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


print('Model defined')

"""###Preprocessing
First we preprocess the data in the same way as for exercise 1 and execute the model **training** to initialize weights for later use. Check that you get more or less the same result as in exercise 1 (mse in the range of 160-162 on the validation data).
"""

#training data preprocess
(x30, g, y) = preprocess_seq(d)
#Validation data preprocess
(x30v, gv, yv) = preprocess_seq(dv)

print(x30[0], g[0], y[0])

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

"""## Exercise 2.1
Masking out input information to estimate its importance for the efficiency prediction.

Below you are presented with ways of masking out the input information in either the sequence or by removing the energy parameter from the model.

The ontarget sequence is stored as a 30mer sequence consisting of

```
4nt prefix + 20nt ontarget + 3nt NGG PAM + 3nt suffix
```

You can for example mask out the first three nucleotides by setting the first 4 values of ```mask``` to 1
```
mask = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
```
or disable the use of the energy parameter by setting
```
use_dgb = False
```


"""

mask=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
use_dgb = True
#training data preprocess
(x30, g, y) = preprocess_seq(d, mask, use_dgb)
#Validation data preprocess
(x30v, gv, yv) = preprocess_seq(dv, mask, use_dgb)

print(x30[0], g[0], y[0])


#restore the initial weights
if os.path.exists('model_30.init.h5'):
  print('weights loaded')
  model.load_weights('model_30.init.h5')
else:
  #did you forget to train the the baseline model above?
  raise Exception

# Training
history = model.fit(
    (x30, g), y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=((x30v,gv), yv),
    callbacks=[es],
    verbose=1
)

# Save the model
print("done")
print("evaulating validation data using best model weights")
model.evaluate((x30v,gv), yv)

"""### Exercise 2.1.1
Modify the code above to run the model without the energy parameter DGb and repeat the training 3-5 times.

What is the effect on the performance?

Is that effect smaller or larger than what is observed for the full model (mse 141.1 with DGb vs 145.1 without). Why would that be?

"""

#answer

"""### Exercise 2.1.2
Modify the code above to mask out part of the ontarget sequence.

What is the effect of masking out the GG of the NGG PAM?

What is the effect of masking out all three nucleotides of the PAM?

What is the effect of masking out first base just before them PAM?

 Does your answers depend on whether DGb is included in the model? For which of the three questions above would it be appropriate / inappropriate to include the energy parameter DGb?

 If time allows, come up with other parts of the sequence to mask out and check your expectation of the impact with the actual result?
"""

# answer

"""## Exercise 2.2

Suggest other features to include in the model, for example epigentic markers or adjusting the model for a particular type of CRISPR experiment?

How would you incorporate them into the model?

What could these features mean for the precision and generalizability of the model?

"""

# answer
