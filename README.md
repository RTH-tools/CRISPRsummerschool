# CRISPRsummerschool 2024

## Exercise 1: Warming up exercise
Introduction to small CRISPR on-target model in Tensorflow / Keras and use it to train a small ontarget efficiency model on real data.

[[Open In GitHub]](https://github.com/RTH-tools/CRISPRsummerschool/tree/main/2024/CRISPR/exercise/crispr_2024_crispr_exercise1.py)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2024/CRISPR/exercise/crispr_2024_crispr_exercise1.ipynb)

## Exercise 2: Extracting features from deep learning
Deep learning does not easily lend itself to extraction of feature importance, like in the example of CRISPR where one could wish to know the importance of e.g. the first nucleotide of the NGG pam for the efficiency of the guide. In this exercise we will look at a way around this problem by masking out parts of the input sequence or of the energy parameter from the model input.

[[Open In GitHub]](https://github.com/RTH-tools/CRISPRsummerschool/tree/main/2024/CRISPR/exercise/crispr_2024_crispr_exercise2.py)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2024/CRISPR/exercise/crispr_2024_crispr_exercise2.ipynb)


## Exercise 3: Convolutions in CRISPR on-target
In this exercise we will take a look at what are the actual outcome of the convolutions of the on-target sequence in the deep learning model.

[[Open In GitHub]](https://github.com/RTH-tools/CRISPRsummerschool/tree/main/2024/CRISPR/exercise/crispr_2024_crispr_exercise3.py)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2024/CRISPR/exercise/crispr_2024_crispr_exercise3.ipynb)

## Exercise 4: A better model?

Create a machine learning model to replace the simple model used in these exercises. The only conditions are

1. Replace the one-hot encoding with an embedding layer
2. Do not use any convolutional layers
3. It should me trainable in less than approximately 5 minutes

This can not be completed by anyone but an expert in the given time, so feel
free to use your favorite Python code generating LLM. You could for example try with
a bidirectional LSTM model. Can you make a model that performs better on the
provided test (validation) data?
