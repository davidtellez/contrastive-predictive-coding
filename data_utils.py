''' This module contains code to handle data '''

import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
from matplotlib import pyplot as plt


class MnistHandler(object):

    ''' Provides a convenient interface to manipulate MNIST data '''

    def __init__(self):

        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()

        # Load Lena image to memory
        self.lena = Image.open('resources/lena.jpg')

    def load_dataset(self):
        # Credit for this function: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

        # We first define a download function, supporting both Python 2 and 3.
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        # We then define functions for loading MNIST images and labels.
        # For convenience, they also download the requested files if needed.
        import gzip

        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the inputs in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            # The inputs are vectors now, we reshape them to monochrome 2D images,
            # following the shape convention: (examples, channels, rows, columns)
            data = data.reshape(-1, 1, 28, 28)
            # The inputs come as bytes, we convert them to float32 in range [0,1].
            # (Actually to range [0, 255/256], for compatibility to the version
            # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
            return data / np.float32(256)

        def load_mnist_labels(filename):
            if not os.path.exists(filename):
                download(filename)
            # Read the labels in Yann LeCun's binary format.
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            # The labels are vectors of integers now, that's exactly what we want.
            return data

        # We can now download and read the training and test set images and labels.
        X_train = load_mnist_images('resources/train-images-idx3-ubyte.gz')
        y_train = load_mnist_labels('resources/train-labels-idx1-ubyte.gz')
        X_test = load_mnist_images('resources/t10k-images-idx3-ubyte.gz')
        y_test = load_mnist_labels('resources/t10k-labels-idx1-ubyte.gz')

        # We reserve the last 10000 training examples for validation.
        X_train, X_val = X_train[:-10000], X_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]

        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test

    def process_batch(self, batch, batch_size, image_size=28, color=False, rescale=True):

        # Resize from 28x28 to 64x64
        if image_size == 64:
            batch_resized = []
            for i in range(batch.shape[0]):
                # resize to 64x64 pixels
                batch_resized.append(scipy.ndimage.zoom(batch[i, :, :], 2.3, order=1))
            batch = np.stack(batch_resized)

        # Convert to RGB
        batch = batch.reshape((batch_size, 1, image_size, image_size))
        batch = np.concatenate([batch, batch, batch], axis=1)

        # Modify images if color distribution requested
        if color:

            # Binarize images
            batch[batch >= 0.5] = 1
            batch[batch < 0.5] = 0

            # For each image in the mini batch
            for i in range(batch_size):

                # Take a random crop of the Lena image (background)
                x_c = np.random.randint(0, self.lena.size[0] - image_size)
                y_c = np.random.randint(0, self.lena.size[1] - image_size)
                image = self.lena.crop((x_c, y_c, x_c + image_size, y_c + image_size))
                image = np.asarray(image).transpose((2, 0, 1)) / 255.0

                # Randomly alter the color distribution of the crop
                for j in range(3):
                    image[j, :, :] = (image[j, :, :] + np.random.uniform(0, 1)) / 2.0

                # Invert the color of pixels where there is a number
                image[batch[i, :, :, :] == 1] = 1 - image[batch[i, :, :, :] == 1]
                batch[i, :, :, :] = image

        # Rescale to range [-1, +1]
        if rescale:
            batch = batch * 2 - 1

        # Channel last
        batch = batch.transpose((0, 2, 3, 1))

        return batch

    def get_batch(self, subset, batch_size, image_size=28, color=False, rescale=True):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Random choice of samples
        idx = np.random.choice(X.shape[0], batch_size)
        batch = X[idx, 0, :].reshape((batch_size, 28, 28))

        # Process batch
        batch = self.process_batch(batch, batch_size, image_size, color, rescale)

        # Image label
        labels = y[idx]

        return batch.astype('float32'), labels.astype('int32')

    def get_batch_by_labels(self, subset, labels, image_size=28, color=False, rescale=True):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Find samples matching labels
        idxs = []
        for i, label in enumerate(labels):

            idx = np.where(y == label)[0]
            idx_sel = np.random.choice(idx, 1)[0]
            idxs.append(idx_sel)

        # Retrieve images
        batch = X[np.array(idxs), 0, :].reshape((len(labels), 28, 28))

        # Process batch
        batch = self.process_batch(batch, len(labels), image_size, color, rescale)

        return batch.astype('float32'), labels.astype('int32')

    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'valid':
            y_len = self.y_val.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]

        return y_len


class MnistGenerator(object):

    ''' Data generator providing MNIST data '''

    def __init__(self, batch_size, subset, image_size=28, color=False, rescale=True):

        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.mnist_handler = MnistHandler()
        self.n_samples = self.mnist_handler.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Get data
        x, y = self.mnist_handler.get_batch(self.subset, self.batch_size, self.image_size, self.color, self.rescale)

        # Convert y to one-hot
        y_h = np.eye(10)[y]

        return x, y_h


class SortedNumberGenerator(object):

    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=28, color=False, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.mnist_handler = MnistHandler()
        self.n_samples = self.mnist_handler.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Build sentences
        image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        positive_samples_n = self.positive_samples
        for b in range(self.batch_size):

            # Set ordered predictions for positive samples
            seed = np.random.randint(0, 10)
            sentence = np.mod(np.arange(seed, seed + self.terms + self.predict_terms), 10)

            if positive_samples_n <= 0:

                # Set random predictions for negative samples
                # Each predicted term draws a number from a distribution that excludes itself
                numbers = np.arange(0, 10)
                predicted_terms = sentence[-self.predict_terms:]
                for i, p in enumerate(predicted_terms):
                    predicted_terms[i] = np.random.choice(numbers[numbers != p], 1)
                sentence[-self.predict_terms:] = np.mod(predicted_terms, 10)
                sentence_labels[b, :] = 0

            # Save sentence
            image_labels[b, :] = sentence

            positive_samples_n -= 1

        # Retrieve actual images
        images, _ = self.mnist_handler.get_batch_by_labels(self.subset, image_labels.flatten(), self.image_size, self.color, self.rescale)

        # Assemble batch
        images = images.reshape((self.batch_size, self.terms + self.predict_terms, images.shape[1], images.shape[2], images.shape[3]))
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]

        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)

        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


class SameNumberGenerator(object):

    ''' Data generator providing lists of similar numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=28, color=False, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.mnist_handler = MnistHandler()
        self.n_samples = self.mnist_handler.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Build sentences
        image_labels = np.zeros((self.batch_size, self.terms + self.predict_terms))
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        positive_samples_n = self.positive_samples
        for b in range(self.batch_size):

            # Set positive samples
            seed = np.random.randint(0, 10)
            sentence = seed * np.ones(self.terms + self.predict_terms)

            if positive_samples_n <= 0:

                # Set random predictions for negative samples
                sentence[-self.predict_terms:] = np.mod(sentence[-self.predict_terms:] + np.random.randint(1, 10, self.predict_terms), 10)
                sentence_labels[b, :] = 0

            # Save sentence
            image_labels[b, :] = sentence

            positive_samples_n -= 1

        # Retrieve actual images
        images, _ = self.mnist_handler.get_batch_by_labels(self.subset, image_labels.flatten(), self.image_size, self.color, self.rescale)

        # Assemble batch
        images = images.reshape((self.batch_size, self.terms + self.predict_terms, images.shape[1], images.shape[2], images.shape[3]))
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]

        # Randomize
        idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)

        return [x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


def plot_sequences(x, y, labels=None, output_path=None):

    ''' Draws a plot where sequences of numbers can be studied conveniently '''

    images = np.concatenate([x, y], axis=1)
    n_batches = images.shape[0]
    n_terms = images.shape[1]
    counter = 1
    for n_b in range(n_batches):
        for n_t in range(n_terms):
            plt.subplot(n_batches, n_terms, counter)
            plt.imshow(images[n_b, n_t, :, :, :])
            plt.axis('off')
            counter += 1
        if labels is not None:
            plt.title(labels[n_b, 0])

    if output_path is not None:
        plt.savefig(output_path, dpi=600)
    else:
        plt.show()


if __name__ == "__main__":

    # Test SortedNumberGenerator
    ag = SortedNumberGenerator(batch_size=8, subset='train', terms=3, positive_samples=4, predict_terms=3, image_size=28, color=True, rescale=False)
    for (x, y), labels in ag:
        plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted.png')
        break

