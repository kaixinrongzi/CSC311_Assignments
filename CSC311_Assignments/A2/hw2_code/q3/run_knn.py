from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int64)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    k_set = []
    accuracy_set_validation = []
    max_accuracy = 0
    optimal_k = 0
    for k in range(1, 10, 2):
        k_set.append(k)
        valid_labels = knn(k, train_inputs, train_targets, valid_inputs)
        assert(len(valid_targets) == len(valid_labels))
        n = len(valid_targets)
        correct = 0
        for i in range(n):
            target = valid_targets[i][0]
            label = valid_labels[i][0]
            if target == label:
                correct += 1
        accuracy = correct / n
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_k = k
        accuracy_set_validation.append(accuracy)

    plt.plot(k_set, accuracy_set_validation, label="Validation accuracy vs. k for KNN Model")
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('model accuracy for validation test')
    plt.title("Validation Accuracy vs. k for KNN Model")
    plt.savefig('Validation accuracy of K for KNN Model.png')
    print("The Maximum accuracy is {0} When k = {1}".format(max_accuracy, optimal_k))

    test_labels = knn(optimal_k, train_inputs, train_targets, test_inputs)
    assert (len(test_targets) == len(test_labels))
    n = len(valid_targets)

    for k in range(optimal_k - 2, optimal_k + 3, 2):
        correct = 0
        for i in range(n):
            target = test_targets[i][0]
            label = test_labels[i][0]
            if target == label:
                correct += 1
        accuracy = correct / n
        print("The Maximum accuracy is {0} When k = {1}".format(accuracy, k))
        print("The Maximum accuracy is {0} When k = {1}".format(accuracy_set_validation[(optimal_k+1)//2 - 1], k))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
