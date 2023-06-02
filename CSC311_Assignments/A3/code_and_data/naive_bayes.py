import numpy as np
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def train_mle_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MLE estimators theta_mle and pi_mle"""

    # YOU NEED TO WRITE THIS PART
    n = train_images.shape[0]
    d = train_images.shape[1]
    map = {}
    for i in range(n):
        data = train_images[i]
        label = train_labels[i]
        c = 0
        while label[c] == 0:
            c += 1
        if c not in map:
            map[c]=[]
        map[c].append(data)

    map2 = {}
    for c in range(10):
        data_lst = map[c]
        # c_num = len(data_lst)
        map3 = {}
        # for data in data_lst:
        #     for i in range(784):
        #         if i not in map3:
        #             map3[i] = 0
        #         map3[i] += data[i]
        map2[c] = np.mean(data_lst, axis=0)
        print(len(map2[c]))
        # for i in range(784):
        #     map3[i] = map3[i] / c_num
        # map2[c] = map3

    theta_mle = []
    for c in range(10):
        frequency = []
        for j in range(784):
            frequency.append(map2[c][j])
        theta_mle.append(frequency)

    theta_mle = np.array(theta_mle)

    pi_mle = []
    for c in range(10):
        pi_mle.append(len(map[c]) / n)
    pi_mle = np.array(pi_mle)


    return theta_mle, pi_mle


def train_map_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MAP estimators theta_map and pi_map"""
    
    # YOU NEED TO WRITE THIS PART
    n = train_images.shape[0]
    d = train_images.shape[1]

    map = {}
    for i in range(n):
        data = train_images[i]
        label_lst = train_labels[i]
        c = 0
        while label_lst[c] == 0:
            c += 1
        if c not in map:
            map[c] = []
        map[c].append(data)

    map1 = {}
    for c in range(10):
        map1[c]=[]
        data_lst = map[c]
        for j in range(d):
            c_num = len(data_lst)
            sum = 0
            for data in data_lst:
                sum += data[j]
            map1[c].append((2+sum)/(4+c_num))

    theta_map = []
    for c in range(10):
        theta_map.append(map1[c])
    theta_map = np.array(theta_map)

    n = theta_map.shape[0]
    d = theta_map.shape[1]

    pi_map = []
    for c in range(n):
        data = theta_map[c]
        pi_map.append(np.sum(data))

    pi_map = np.array(pi_map)

    return theta_map, pi_map


def log_likelihood(images, theta, pi):
    """ Inputs: images, theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
    log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
    log_like is a matrix of num of images x num of classes
    Note that log likelihood is not only for c^(i), it is for all possible c's."""

    # YOU NEED TO WRITE THIS PART
    n = images.shape[0]
    d = images.shape[1]
    # map = {}
    # for c in range(10):
    #     map[c] = []
    #
    #     for i in range(n):
    #         data = images[i]
    #         log_p = 0
    #         for j in range(d):
    #             log_p += data[j] * np.log(theta[c][j]) + (1-data[j])*np.log(1-theta[c][j])
    #             print(i, j)
    #         log_p += np.log(pi[c])
    #         log_p -= np.log(1/n)
    #
    #         map[c].append(log_p)
    map = {}
    for i in range(n):
        print(i)
        data = images[i]
        map[i]={}
        for c in range(10):
            map[i][c]=0
            for j in range(d):
                map[i][c] += data[j]*theta[c][j]+(1-data[j])*np.log(1-theta[c][j])
            map[i][c] += pi[c]
            map[i][c] -= np.log(1/n)

    log_like = []
    for i in range(n):
        p_lst_map = map[i]
        p_lst = []
        for c in range(10):
            p_lst.append(p_lst_map[c])
        log_like.append(p_lst)

    log_like = np.array(log_like)

    # for c in range(10):
    #     log_like.append(map[c])
    # log_like = np.array(log_like)

    return log_like


def predict(log_like):
    """ Inputs: matrix of log likelihoods
    Returns the predictions based on log likelihood values"""

    # YOU NEED TO WRITE THIS PART
    n = log_like.shape[0]
    c_num = log_like.shape[1]

    predictions = []
    for i in range(n):
        log_p_lst = log_like[i]
        max_log_p = -np.infty
        optimal_c = 0
        for c in range(c_num):
            log_p = log_p_lst[c]
            if log_p > max_log_p:
                max_log_p = log_p
                optimal_c = c
        predictions.append(optimal_c)

    predictions = np.array(predictions)

    return predictions


def accuracy(log_like, labels):
    """ Inputs: matrix of log likelihoods and 1-of-K labels
    Returns the accuracy based on predictions from log likelihood values"""

    # YOU NEED TO WRITE THIS PART
    predictions = predict(log_like)

    n = predictions.shape[0]
    total_accurate_num = 0
    for i in range(n):
        pred = predictions[i]
        label_lst = labels[i]
        label = 0
        while label_lst[label] == 0:
            label += 1
        total_accurate_num += (pred==label)

    acc = total_accurate_num / float(n)

    return acc


def main():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    # Fit MLE and MAP estimators
    theta_mle, pi_mle = train_mle_estimator(train_images, train_labels)
    theta_map, pi_map = train_map_estimator(train_images, train_labels)

    # Find the log likelihood of each data point
    loglike_train_mle = log_likelihood(train_images, theta_mle, pi_mle)
    loglike_train_map = log_likelihood(train_images, theta_map, pi_map)

    avg_loglike_mle = np.sum(loglike_train_mle * train_labels) / N_data
    avg_loglike_map = np.sum(loglike_train_map * train_labels) / N_data

    print("Average log-likelihood for MLE is ", avg_loglike_mle)
    print("Average log-likelihood for MAP is ", avg_loglike_map)

    train_accuracy_map = accuracy(loglike_train_map, train_labels)
    loglike_test_map = log_likelihood(test_images, theta_map, pi_map)
    test_accuracy_map = accuracy(loglike_test_map, test_labels)

    print("Training accuracy for MAP is ", train_accuracy_map)
    print("Test accuracy for MAP is ", test_accuracy_map)

    # Plot MLE and MAP estimators
    save_images(theta_mle, 'mle.png')
    save_images(theta_map, 'map.png')


if __name__ == '__main__':
    main()
