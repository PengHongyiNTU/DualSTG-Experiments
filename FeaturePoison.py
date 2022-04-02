"""
======================= START OF LICENSE NOTICE =======================
  Copyright (C) 2022 HONGYI001. All Rights Reserved

  NO WARRANTY. THE PRODUCT IS PROVIDED BY DEVELOPER "AS IS" AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DEVELOPER BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE PRODUCT, EVEN
  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
======================== END OF LICENSE NOTICE ========================
  Primary Author: HONGYI001

"""
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
    X_noise = X.copy()
    if num_noise_features == 0:
        return X_noise
    rng = np.random.default_rng(seed)
    if noise_type == "normal":
        for i in range(num_noise_features):
            X_noise["Noise-{}".format(i)] = rng.normal(0, noise_std, X.shape[0])
    else:
        for i in range(num_noise_features):
            # print(X.shape[0])
            # print(X_noise.shape[0])
            # print(skewnorm.rvs(a=noise_skewness, size=X.shape[0]))
            X_noise["Skewed-Noise-{}".format(i)] = skewnorm.rvs(
                a=noise_skewness, size=X.shape[0])
            X_noise["Noise-{}".format(i)] = np.random.normal(0, noise_std, X.shape[0])
    # X_noise = pd.DataFrame(MinMaxScaler().fit_transform(X_noise), columns=X_noise.columns)
    return X_noise


def insert_overwhelmed_noise(
    X,
    num_noise_features,
    noise_lambda_range=(0, 5),
    coefficient_range=(-10, 10),
    seed=0,
):
    X_noise = X.copy()
    if num_noise_features == 0:
        return X_noise
    rng = np.random.default_rng(seed)
    for i in range(num_noise_features):
        feature = "Overwhelmed Feature " + str(i + 1)
        idxs = rng.choice(X.shape[1], size=rng.integers(1, X.shape[1]), replace=False)
        X_noise[feature] = np.zeros(X.shape[0])
        for idx in idxs:
            coefficient = rng.uniform(*coefficient_range)
            X_noise[feature] += coefficient * X.iloc[:, idx]
        noise = rng.normal(0.5, 0.25, X_noise.shape[0])
        X_noise[feature] = X_noise[feature] + rng.uniform(*noise_lambda_range) * noise
    return X_noise


def insert_shortcut_noise(X, y,
    num_noise_features=1, ps=[0.3], 
    seed=0, 
    test_size=0.2):
    if num_noise_features == 0:
        return train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    rng = np.random.default_rng(seed)
    assert len(ps) == num_noise_features
    X_noise = X.copy()
    for i in range(num_noise_features):
        p = ps[i]
        X_noise["Short Cut Feature" + str(i) + "-" + str(p)] = rng.choice(
            np.unique(y), X.shape[0]
        ) + rng.normal(0.5, 1, X.shape[0])
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_noise, y, test_size=test_size, random_state=seed
    )
    label = np.unique(y)
    dummy_fill = np.vectorize(lambda x: rng.choice(np.delete(label, x), 1))
    for i in range(num_noise_features):
        p = ps[i]
        X_train["Short Cut Feature" + str(i) + "-" + str(p)].iloc[
            : int(p * X.shape[0])
        ] = Y_train[: int(p * X.shape[0])]
        X_test["Short Cut Feature" + str(i) + "-" + str(p)].iloc[
            : int((1 - p) * X.shape[0])
        ] = dummy_fill(Y_test[: int((1 - p) * X.shape[0])])
    return X_train, X_test, Y_train, Y_test


def insert_feature_noise(X, y, 
    num_random_noise=10, num_overwhelmed=5, num_shortcut=5,
    noise_std=0.1, noise_skewness=3, noise_type="both",
    noise_lambda_range=(0, 5), 
    coefficient_range=(-10, 10),
    p=0.3, test_size=0.2, 
    seed=0):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    pd.set_option('mode.chained_assignment', None)
    X_noise = insert_random_noise(
        X, num_random_noise, noise_std, noise_skewness, noise_type, seed
    )
    X_noise = insert_overwhelmed_noise(
        X_noise, num_overwhelmed, noise_lambda_range, coefficient_range, seed
    )
    X_train, X_test, Y_train, Y_test = insert_shortcut_noise(
        X_noise, y, num_shortcut, [p]*num_shortcut, seed, test_size
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    x = pd.DataFrame(np.random.randn(100, 10))
    X_train, X_test, Y_train, Y_test = insert_feature_noise(x, np.random.randint(0, 2, size=100))
    print(X_train.shape)
