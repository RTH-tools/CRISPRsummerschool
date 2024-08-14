# -*- coding: utf-8 -*-
"""crispr_2024_crispr_exercise4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dC1-BS5l180ZA9a-eAarSb0ehEmIG4yn

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2024/CRISPR/exercise/crispr_2024_crispr_exercise4.ipynb)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, SimpleRNN, Dense, Dropout, Input, Attention, Add
from tensorflow.keras.optimizers import Adam

! [[ -e training_data.csv ]] || curl -o training_data.csv https://rth.dk/internal/index.php/s/hJ45ZdUExZyZ6xe/download
! [[ -e validation_data.csv ]] || curl -o validation_data.csv https://rth.dk/internal/index.php/s/lhufyeBtc8BA3K8/download

# Example sequence-to-integer mapping
def encode_sequence(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return [mapping[nucleotide] for nucleotide in seq]

# Load your dataset
data = pd.read_csv('training_data.csv')
val_data = pd.read_csv('validation_data.csv')

# Encode the 30-mer sequences
X_train = np.array([encode_sequence(seq) for seq in data['target']])
y_train = data['eff'].values

X_val = np.array([encode_sequence(seq) for seq in val_data['target']])
y_val = val_data['eff'].values


# Split the data into training and test sets

# Convert to numpy arrays
X_train = np.array(X_train)
X_val = np.array(X_val)

# Parameters
vocab_size = 4  # Number of unique nucleotides (A, C, G, T)
embedding_dim = 32  # Size of the embedding vector
input_length = 30  # Length of each sequence (30-mer)
rnn_units = 64  # Number of units in the RNN layer

# Input layer
inputs = Input(shape=(input_length,))

# Embedding layer
x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(inputs)

# Bidirectional RNN layer
x = Bidirectional(LSTM(units=rnn_units, return_sequences=True))(x)

# Self-Attention mechanism
attention = Attention()([x, x])
x = Add()([x, attention])

# Pooling the output of the RNN + Attention to feed into the Dense layer
x = LSTM(units=rnn_units, return_sequences=False)(x)

# Dropout layer for regularization
x = Dropout(0.2)(x)

# Dense output layer
x = Dense(120, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(120, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)
# Model definition
model = Model(inputs=inputs, outputs=[outputs, attention])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Model summary
model.summary()

es = callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.1,
            patience=10,
            verbose=1,
            mode='auto',
            restore_best_weights=True,
            ),

# Training the model

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32,     callbacks=[es],)

# Predict on the test set
y_pred, attention_weights = model.predict(X_val)

# Calculate Mean Squared Error
mse = mean_squared_error(y_val, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

print(attention_weights.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# Example: Visualizing attention weights for the first test sample
for sample_index in range(20):
  sequence = X_val[sample_index]
  weights = attention_weights[sample_index]

  # Plot attention weights
  plt.figure(figsize=(10, 2))
  sns.heatmap(weights, cmap='viridis')
  plt.title(f'Attention Weights for Sample {sample_index} with eff {y_pred[sample_index]}')
  plt.xlabel('Sequence Position')
  plt.ylabel('Attention Weights')
  plt.show()
