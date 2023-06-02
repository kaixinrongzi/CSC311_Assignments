'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    map = {}
    for k in range(10):
        map[k] = []

    n = len(train_labels)
    for i in range(n):
        label = train_labels[i]
        data = train_data[i]
        map[label].append(data)

    for k in range(10):
        lst = np.array(map[k]).reshape(len(map[k]), 64)
        map[k] = np.mean(lst, axis=0)

    mean_lst = []
    for k in range(10):
        print(len(map[k]))
        mean_lst.append(map[k])

    means = np.array(mean_lst)

    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    map = {}
    for k in range(10):
        map[k] = []

    means = compute_mean_mles(train_data, train_labels)
    n = len(train_labels)
    for i in range(n):
        label = int(train_labels[i])
        data = train_data[i]
        map[label].append(data)

    sum_map = {}
    for k in range(10):
        data_lst = map[k]
        sum_map[k] = []
        for data in data_lst:
            diff = data - means[k]
            diff_mtx = np.array(diff).reshape(64, 1)
            diff_mtx_transpose = np.array(diff).reshape(1, 64)
            res = np.matmul(diff_mtx, diff_mtx_transpose)
            sum_map[k].append(res)

    avg_map = {}
    for k in range(10):
        matrices = sum_map[k]
        num_matrices = len(matrices)
        matrix_sum = np.zeros((64, 64))
        for matrix in matrices:
            matrix_sum += matrix
        avg_map[k] = matrix_sum / num_matrices

    avg_lst = []
    for k in range(10):
        avg_lst.append(avg_map[k])

    covariances = np.array(avg_lst)

    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    # assuming there is no relationship between pixel -> modify the covariance
    # new_covariances = []
    # for k in range(10):
    #     cov_k = covariances[k]
    #     cov_k = np.diag(np.diag(cov_k))
    #     new_covariances.append(cov_k)
    # covariances = np.array(new_covariances)

    n = len(digits)
    res = []
    for i in range(n):
        pixels = digits[i]
        p_lst = []
        for k in range(10):
            mu_k = means[k]
            cov_k = covariances[k]
            cov_k = np.add(cov_k, 0.01 * np.identity(64))    # add 0.01I for stabability

            det = np.linalg.det(cov_k)

            shit2 = det**(-0.5)
            # if shit2 == 0:
            #     print("shit2 == 0")
            shit3 = (2*np.pi)**(-32)
            # if shit3 == 0:
            #     print("shit3 == 0")

            diff = (pixels - mu_k).reshape(64, 1)
            diff_transpose = (pixels - mu_k).reshape(1, 64)

            shit_shit = -0.5*np.matmul(np.matmul(diff_transpose, np.linalg.inv(cov_k)), diff)
            shit = np.exp(shit_shit)
            # if shit == 0:
            #     print("shit == 0", "k={}, i={}".format(k, i))

            log_p = np.log(shit3*shit2*shit)

            p_lst.append(log_p[0][0])

        res.append(p_lst)

    res = np.array(res)

    return res


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''

    # get log p(x | y)
    generative_probs = generative_likelihood(digits, means, covariances)

    n = len(generative_probs)  # n = 7000
    res = []
    for i in range(n):
        p_lst = generative_probs[i]
        conditional_log_p_lst = []
        for log_p in p_lst:
            conditional_log_p = log_p + np.log(0.1) - np.log(1/n)
            conditional_log_p_lst.append(conditional_log_p)
        res.append(conditional_log_p_lst)

    res = np.array(res)

    return res


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    n = len(cond_likelihood)
    total_p = 0
    temp = 0
    for i in range(n):
        label = int(labels[i])
        p_lst = cond_likelihood[i]
        p = p_lst[label]
        if i == 0:
            temp = p
        elif i == 1:
            total_p = np.logaddexp(temp, p)
        else:
            total_p = np.logaddexp(total_p, p)

    return total_p / float(n)


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    n = len(cond_likelihood)
    res = []
    for i in range(n):
        log_p_lst = cond_likelihood[i]
        optimal_label = 0
        max_log_p = - np.infty
        for k in range(10):
            if log_p_lst[k] > max_log_p:
                optimal_label = k
                max_log_p = log_p_lst[k]
        res.append(optimal_label)

    res = np.array(res)

    return res


def get_accuracy(digits, labels, means, covariances):
    optimal_labels_lst = classify_data(digits, means, covariances)
    assert len(optimal_labels_lst) == len(labels)
    n = len(optimal_labels_lst)
    total_accurate_num = 0
    for i in range(n):
        total_accurate_num += (optimal_labels_lst[i] == labels[i])

    return total_accurate_num / float(n)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    train_res = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_res = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print("train_res: ", train_res)
    print('test_res: ', test_res)

    # Get accurary
    print(get_accuracy(train_data, train_labels, means, covariances))
    print(get_accuracy(test_data, test_labels, means, covariances))



if __name__ == '__main__':
    main()
