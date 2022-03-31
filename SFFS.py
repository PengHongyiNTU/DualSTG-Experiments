import numpy as np
from numpy.linalg import inv, pinv
from time import time

# from Data import VFLDataset



def get_theta_param(data_x, data_y):
    corvariance_x = np.matmul(data_x.T, data_x)

    pinv_begin = time()
    inv_corvariance_x = pinv(corvariance_x)
    pinv_end = time()

    x_transpose_y = np.matmul(data_x.T, data_y)

    return corvariance_x, np.matmul(inv_corvariance_x, x_transpose_y), pinv_end - pinv_begin


def get_f_statistics(data_x, data_y):
    feature_num = data_x.shape[1]
    f_statistics = np.zeros(feature_num)

    corvariance_x, theta_param, computation_time= get_theta_param(data_x, data_y)
    diag_x = np.diag(corvariance_x)

    for j in range(feature_num):
        f_statistics[j] = theta_param[j] ** 2 / diag_x[j]

    print('total computation time for pinv is:', computation_time)

    return f_statistics


def get_f_stat_index(data_X, data_y):

    f_stat = get_f_statistics(data_X, data_y)
    index = np.argsort(-f_stat)

    return index



if __name__ == "__main__":
    pass

    # data_set = VFLDataset("madelon", scale=True, num_clients=1)

    # index = get_f_stat_index(data_set)
    # chosen_index = index[int(index.shape[0]/2):]

    # sffs_data_set = VFLDataset("madelon", scale=True, num_clients=2, sffs=chosen_index)



















