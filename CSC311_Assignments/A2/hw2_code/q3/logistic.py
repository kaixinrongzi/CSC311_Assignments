import math
import random
from copy import deepcopy
from shutil import copy

from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    y = []

    # 1. reshape data
    n = len(data)
    data_cp = deepcopy(data)
    data_cp = data_cp.tolist()
    for i in range(n):
        data_cp[i].append(1)

    # 2.  calculate z
    weights_array = np.array(weights)
    data_cp_array = np.array(data_cp)

    z = np.matmul(data_cp_array, weights_array)
    z2 = z * (-1)

    # 3. calculate y = 1 / 1 + e^(-z)
    assert(len(z2) == len(data_cp))
    out = np.exp(z2)
    assert (len(z2) == len(out))
    n = len(out)
    for i in range(n):
        y.append([1 / (out[i][0] + 1)])
    y = np.array(y)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    ce = 0
    frac_correct = 0
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    assert(len(targets) == len(y))

    n = len(targets)
    total_crossEntropy_loss = 0
    correct = 0
    for i in range(n):
        t = targets[i][0]
        _y = y[i][0]
        total_crossEntropy_loss += -t * math.log2(_y) - (1-t) * math.log2(1-_y)

        if abs(t - _y) < .1:
            correct += 1

    ce = total_crossEntropy_loss / n
    frac_correct = correct / n

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    f = None
    df = []

    # 1. reshape data
    n = len(data)
    data_cp = deepcopy(data)
    data_cp = data_cp.tolist()
    for i in range(n):
        data_cp[i].append(1)

    # 2.  calculate z
    weights_array = np.array(weights)
    data_cp_array = np.array(data_cp)

    z = np.matmul(data_cp_array, weights_array)
    assert(len(z) == len(y))
    n = len(y)

    # 3. calculate derivative for w_i, 1 <= i <= M

    assert(len(weights) == len(data_cp[0]))
    m = len(weights)

    # randomly choose <hyperparamters> numbers of data points from data
    # if hyperparameters > n:
    #     data_set = data_cp
    # else:
    #     data_set = []
    #     chosen_random_index = []
    #     chosen = 0
    #     while chosen < hyperparameters:
    #         random_index = random.randint(0, n-1)
    #         if random_index in chosen_random_index:
    #             continue
    #         chosen_random_index.append(random_index)
    #         data_set.append(data_cp[random_index])
    #         chosen += 1
    data_set = data_cp

    for i in range(m):
        total_loss = 0
        derivative = 0
        for j in range(n):
            prob = y[j][0]
            target = targets[j][0]
            total_loss += (-1) * target * np.log2(prob) - (1 - target) * np.log2(1 - prob)

            derivative1 = (target - prob)/((prob - 1) * prob * math.log(2))
            derivative2 = math.exp(-z[j][0]) / (math.exp(-z[j][0]) + 1) ** 2
            derivative3 = data_set[j][i]

            derivative += derivative1 * derivative2 * derivative3

        f = total_loss / n
        derivative = derivative / n
        df.append([derivative])

    df = np.array(df)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


if __name__ == "__main__":
    logistic([0.1, 0.2, 0.3, 0.4, 0.25], [[1, 2, 3, 4], [4, 2, 6, 7], [3, 1, 9, 2], [3, 3, 2, 2]], [1, 1, 1], 3)
