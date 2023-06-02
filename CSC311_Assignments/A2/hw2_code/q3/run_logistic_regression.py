from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 500
    }
    weights = np.random.randn(M + 1, 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    new_weights = weights
    cross_entropy_set = []
    x = []
    y = None
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(new_weights, valid_inputs, valid_targets, hyperparameters)
        # use df to find out to converge weights
        learning_rate = hyperparameters["learning_rate"]
        temp_df = learning_rate * df
        new_weights = np.subtract(new_weights, temp_df)
        # print(evaluate(train_targets, y))
        cross_entropy_set.append(evaluate(valid_targets, y)[0])
        x.append(t)
    plt.plot(x, cross_entropy_set, label="cross_entropy changes as training progress using validation_set")
    print("E(J(y, t)) of validation data = " + str(np.std(y) + np.std(valid_targets)))

    new_weights = weights
    cross_entropy_set = []
    x = []
    y = None
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(new_weights, train_inputs, train_targets, hyperparameters)
        # use df to find out to converge weights
        learning_rate = hyperparameters["learning_rate"]
        temp_df = learning_rate * df
        new_weights = np.subtract(new_weights, temp_df)
        # print(evaluate(train_targets, y))
        cross_entropy_set.append(evaluate(train_targets, y)[0])
        x.append(t)
    plt.plot(x, cross_entropy_set, label="cross_entropy changes as training progress using mnist_train_small")
    print("E(J(y, t)) of training data = " + str(np.std(y) + np.std(train_targets)))

    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("cross_entropy")
    plt.title("cross_entropy changes as training progress")
    plt.savefig("cross_entropy changes as training progress")



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff=", diff)


if __name__ == "__main__":
    run_logistic_regression()
