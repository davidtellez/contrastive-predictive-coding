### Representation Learning with Contrastive Predictive Coding

This repository contains a Keras implementation of the algorithm presented in the paper [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748).

The goal of unsupervised representation learning is to capture semantic information about the world, recognizing patterns in the data without using annotations. This paper presents a new method called Contrastive Predictive Coding (CPC) that can do so across multiple applications. The main ideas of the paper are:
* Contrastive: it is trained using a contrastive approach, that is, the main model has to discern between *right* and *wrong* data sequences.
* Predictive: the model has to predict future patterns given the current context.
* Coding: the model performs this prediction in a latent space, transforming code vectors into other code vectors (in contrast with predicting high-dimensional data directly).

CPC has to predict the next item in a sequence using only an embedded representation of the data, provided by an encoder. In order to solve the task, this encoder has to learn a meaningful representation of the data space. After training, this encoder can be used for other downstream tasks like supervised classification.

<p align="center">
<img src="/resources/figure.png" alt="CPC algorithm" height="350">
</p>

To train the CPC algorithm, I have created a toy dataset. This dataset consists of sequences of modified MNIST numbers (64x64 RGB). Positive sequence samples contain *sorted* numbers, and negative ones *random* numbers. For example, let's assume that the context sequence length is S=4, and CPC is asked to predict the next P=2 numbers. A positive sample could look like ```[2, 3, 4, 5]->[6, 7]```, whereas a negative one could be ```[1, 2, 3, 4]->[0, 8]```. Of course CPC will only see the patches, not the actual numbers.

Disclaimer: this code is provided *as is*, if you encounter a bug please report it as an issue. Your help will be much welcomed!

### Results

After 10 training epochs, CPC reports a 99% accuracy on the contrastive task. After training, I froze the encoder and trained a MLP on top of it to perform supervised digit classification on the same MNIST data. It achieved 90% accuracy after 10 epochs, demonstrating the effectiveness of CPC for unsupervised feature extraction.

### Usage

- Execute ```python train_model.py``` to train the CPC model.
- Execute ```python benchmark_model.py``` to train the MLP on top of the CPC encoder.

### Requisites

- [Anaconda Python 3.5.3](https://www.continuum.io/downloads)
- [Keras 2.0.6](https://keras.io/)
- [Tensorflow 1.4.0](https://www.tensorflow.org/)
- GPU for fast training.

### References

- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
