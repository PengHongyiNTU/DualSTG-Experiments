import numpy as np
from scipy.stats import skewnorm
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import MinMaxScaler

def insert_random_noise(
    X, num_noise_features=5, 
    noise_std=0.1, 
    noise_skewness=3, 
    noise_type="both", 
    seed=0
): 
    if noise_type == "normal":
        noise_mat = np.zeros((X.shape[0], num_noise_features))
    else:
        noise_mat = np.zeros((X.shape[0], 2*num_noise_features))
    rng = np.random.default_rng(seed)
    if noise_type == "normal":
        for i in range(num_noise_features):
            noise_mat[:, i] = rng.normal(0, noise_std, X.shape[0])
    else:
        for i in range(num_noise_features):
            noise_mat[:, i] = skewnorm.rvs(a=noise_skewness, size=X.shape[0])
            noise_mat[:, i+num_noise_features] = rng.normal(0, noise_std, X.shape[0])
    return np.concatenate([X, noise_mat], axis=1)

def insert_overwhelmed_noise(
    X,
    num_noise_features,
    noise_lambda_range=(0, 5),
    coefficient_range=(-10, 10),
    seed=0,
):
    if num_noise_features == 0:
        return X
    noise_mat = np.zeros((X.shape[0], num_noise_features))
    rng = np.random.default_rng(seed)
    for i in range(num_noise_features):
        idxs = rng.choice(X.shape[1], size=rng.integers(1, X.shape[1]), replace=False)
        noise_mat[:, i] = np.zeros(X.shape[0])
        for idx in idxs:
            coefficient = rng.uniform(*coefficient_range)
            noise_mat[:, i] += coefficient * X[:, idx]
        noise = rng.normal(0.5, 0.25, X.shape[0])
        noise_mat[:, i] = noise_mat[:, i] + rng.uniform(*noise_lambda_range) * noise
    return np.concatenate([X, noise_mat], axis=1)

def insert_shortcut_noise(X, y,
    num_noise_features=10, ps=[0.3], 
    seed=0, 
    test_size=0.2):
    if num_noise_features == 0:
        return X
    rng = np.random.default_rng(seed)
    assert len(ps) == num_noise_features
    noise_mat = np.random.randn(X.shape[0], num_noise_features)
    # print(noise_mat.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)    
    train_len = X_train.shape[0]
    test_len = X_test.shape[0]
    label = np.unique(y)
    dummy_fill = np.vectorize(lambda x: rng.choice(np.delete(label, x), 1))
    for i in range(num_noise_features):
        p = ps[i]
        split = int(p*train_len)
        #print(noise_mat[i].shape)
        noise_mat[:split, i] = y_train[:split]
        # print(noise_mat[:,i])
        split = int(p*test_len)
        # print(temp)
        # print(y_test[:-split])
        noise_mat_test = noise_mat[:-train_len]
        noise_mat_test[:-split:, i] = dummy_fill(y_test[:-split])
    X_train = np.concatenate([X_train, noise_mat[:train_len, :]], axis=1)
    X_test = np.concatenate([X_test, noise_mat[:-train_len, :]], axis=1)
    return X_train, X_test, y_train, y_test



def insert_feature_noise(X, y, 
    num_random_noise=10, num_overwhelmed=5, num_shortcut=5,
    noise_std=0.1, noise_skewness=3, noise_type="both",
    noise_lambda_range=(0, 5), 
    coefficient_range=(-10, 10),
    p=0.3, test_size=0.2, 
    seed=0):
    X_noise = insert_random_noise(X, num_random_noise, noise_std, noise_skewness, noise_type, seed)
    X_noise = insert_overwhelmed_noise(X_noise, num_overwhelmed, noise_lambda_range, coefficient_range, seed)
    X_train, X_test, y_train, y_test = insert_shortcut_noise(X_noise, y, num_shortcut, [p]*num_shortcut, seed, test_size)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X = np.random.randn(100, 10)
    y = np.concatenate([np.zeros((50, )), np.ones((50, ))], axis=0)
    y = y.astype(int)
    X_train, X_test, y_train, y_test = insert_feature_noise(X, y)
    print(X_train.shape)