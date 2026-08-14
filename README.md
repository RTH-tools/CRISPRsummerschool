# CRISPRsummerschool 2026

## Exercise 1: Introduction
Introduction to a small CRISPR on-target model in PyTorch and use it to train a small ontarget efficiency model on real data. One-hot encoding, mini-batches and epochs, early stopping, and evaluation on validation data. The solutions will be available after the class.

[[Open In GitHub]](https://github.com/RTH-tools/CRISPRsummerschool/tree/main/2026/CRISPR/exercise/crispr_2026_crispr_exercise1.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2026/CRISPR/exercise/crispr_2026_crispr_exercise1.ipynb)

## Exercise 2: Extracting features from deep learning
Deep learning does not easily lend itself to extraction of feature importance, like in the example of CRISPR where one could wish to know the importance of e.g. the first nucleotide of the NGG pam for the efficiency of the guide. In this exercise we will look at a way around this problem by masking out parts of the input sequence or of the energy parameter from the model input.

[[Open In GitHub]](https://github.com/RTH-tools/CRISPRsummerschool/tree/main/2026/CRISPR/exercise/crispr_2026_crispr_exercise2.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2026/CRISPR/exercise/crispr_2026_crispr_exercise2.ipynb)

## Exercise 3: Convolutions in CRISPR on-target
In this exercise we will take a look at what are the actual outcome of the convolutions of the on-target sequence in the deep learning model. We inspect the shape of the convolution output, print the learned filter weights, and apply a hand-set filter of our own to see what a single convolution responds to.

[[Open In GitHub]](https://github.com/RTH-tools/CRISPRsummerschool/tree/main/2026/CRISPR/exercise/crispr_2026_crispr_exercise3.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2026/CRISPR/exercise/crispr_2026_crispr_exercise3.ipynb)

## Exercise 4: CRISPRon with uncertainty estimation
A model that outputs a single number hides something important: how sure is it? Two guides can both be predicted at 63% efficiency, yet the model may be confident about one and essentially guessing about the other, i.e. uncertain. In this exercise we change the output head so that the model predicts a mean and a variance, train it with the Gaussian negative log-likelihood instead of the mean squared error, and add a five-member deep ensemble.

That separates two kinds of uncertainty: **aleatoric**, the noise inherent in the data, which more data does not remove, and **epistemic**, the model's own ignorance about guides unlike anything it was trained on, which more data does help. We then ask the harder question - whether the reported uncertainty can be trusted - by checking whether it tracks the actual error and whether it is calibrated.

[[Open In GitHub]](https://github.com/RTH-tools/CRISPRsummerschool/tree/main/2026/CRISPR/exercise/crispr_2026_crispr_exercise4.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RTH-tools/CRISPRsummerschool/blob/main/2026/CRISPR/exercise/crispr_2026_crispr_exercise4.ipynb)


## Running on Google Colab

The Google Colab links above open a temporary session that does not autosave. Before anything, save your own copy and enable the GPU. Note that re-clicking a link always reopens the original.

1. **File → Save a copy in Drive**
2. **Runtime → Change runtime type**
3. **Hardware accelerator → T4 GPU**
4. **Save** (the session restarts)

Confirm it worked: run the first cell and read the device it prints.

```
Using device: cuda      <- GPU active
Using device: cpu       <- still on CPU
```
