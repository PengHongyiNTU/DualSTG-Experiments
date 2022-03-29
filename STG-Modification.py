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
import warnings
import pandas as pd
import ScikitWrapper as skw
from Gini import gini_filter
import torch
import numpy as np
from FeaturePoison import insert_feature_noise
import shap
import seaborn as sns
import matplotlib.pyplot as plt



def print_accuracy(f, X, y):
    accuracy = 100*np.sum(f(X) == y)/len(y)
    print("Accuracy = {0}%".format(accuracy))
    return accuracy

num_normal = np.arange(0, 15, 1)
num_overwhelmed = [3]*len(num_normal)
num_shortcut = [1, 2, 3]
num_shortcut = np.repeat(np.array(num_shortcut), 5)
warnings.simplefilter(action='ignore')
X, y = shap.datasets.iris()
no_stg_results = {
    '# features': [],
    "accuracy": []
}

gini_half_results = {
    '# features': [],
    "accuracy": []
}

with_stg_results = {
    '# features': [],
    "accuracy": []
}
two_step_results = {
    '# features': [],
    "accuracy": []
}

for t in range(5):
    accs = []
    for i in range(len(num_normal)):
        X_train, X_test, Y_train, Y_test = insert_feature_noise(X, y, num_random_noise=num_normal[i],
            num_overwhelemed=num_overwhelmed[i], num_shortcut=num_shortcut[i])
        X_train, y_train = X_train.to_numpy().astype(np.float32), Y_train.astype(np.int64)
        X_test, y_test = X_test.to_numpy().astype(np.float32), Y_test.astype(np.int64)
        clf = skw.ScikitClfWrapper(
        input_dim=X_train.shape[1],
        number_of_classes=3,
        hidden_dims=(16, 16), lam=0, epochs=1000, sigma=0.5,
        freeze_till=1000, lr=0.1, verbose=False,
        device='cpu')
        clf.fit(X_train, Y_train)
        acc = print_accuracy(clf.predict, X_test, Y_test)
        no_stg_results["accuracy"].append(acc)
        total_feature = 2*num_normal[i] + num_overwhelmed[i] + num_shortcut[i]
        no_stg_results['# features'].append(total_feature)
        print(f"{acc}%, {total_feature+4} features")




for t in range(5):
    accs = []
    for i in range(len(num_normal)):
        X_train, X_test, Y_train, Y_test = insert_feature_noise(X, y, num_random_noise=num_normal[i],
            num_overwhelemed=num_overwhelmed[i], num_shortcut=num_shortcut[i])
        

        X_train, X_test, Y_train, Y_test = gini_filter(torch.tensor(X_train.values), torch.tensor(X_test.values), 
            torch.tensor(Y_train, dtype=torch.int64), 
            torch.tensor(Y_test, dtype=torch.int64), 
            left=0.5)

        X_train, X_test, Y_train, Y_test = X_train.numpy(), X_test.numpy(), Y_train.numpy(), Y_test.numpy()
        X_train, Y_train = X_train.astype(np.float32), Y_train.astype(np.int64)
        X_test, Y_test = X_test.astype(np.float32), Y_test.astype(np.int64)
        """
        clf = FNNClfWrapper(
        input_dim=X_train.shape[1],
        number_of_classes=3,
        hidden_dims=(16, 16), epochs=500, lr=1e-3, verbose=True,
        device='cpu', batch_norm=True, dropout=0.5)
        """
        
        clf = skw.ScikitClfWrapper(
        input_dim=X_train.shape[1],
        number_of_classes=3,
        hidden_dims=(16, 16), lam=0, epochs=1000, sigma=1.0,
        freeze_till=1000, lr=0.1, verbose=False,
        device='cpu')
        # print(X_train.shape)
        
        clf.fit(X_train, Y_train)
        acc = print_accuracy(clf.predict, X_test, Y_test)
        gini_half_results["accuracy"].append(acc)
        total_feature = 2*num_normal[i] + num_overwhelmed[i] + num_shortcut[i]
        gini_half_results['# features'].append(total_feature)
        print(f"{acc}%, {total_feature} features")



for t in range(5):
    accs = []
    for i in range(len(num_normal)):
        X_train, X_test, Y_train, Y_test = insert_feature_noise(X, y, num_random_noise=num_normal[i],
            num_overwhelemed=num_overwhelmed[i], num_shortcut=num_shortcut[i])
        X_train, y_train = X_train.to_numpy().astype(np.float32), Y_train.astype(np.int64)
        X_test, y_test = X_test.to_numpy().astype(np.float32), Y_test.astype(np.int64)
        clf = skw.ScikitClfWrapper(
        input_dim=X_train.shape[1],
        number_of_classes=3,
        hidden_dims=(16, 16), lam=0.1, epochs=1000, sigma=0.5,
        freeze_till=0, lr=0.1, verbose=False,
        device='cpu')
        clf.fit(X_train, Y_train)
        acc = print_accuracy(clf.predict, X_test, Y_test)
        with_stg_results["accuracy"].append(acc)
        total_feature = 2*num_normal[i] + num_overwhelmed[i] + num_shortcut[i]
        with_stg_results['# features'].append(total_feature)
        _, num_gates = clf.model.get_gates()
        print(f"{acc}%, {num_gates}/{total_feature+4} features")




for t in range(5):
    accs = []
    for i in range(len(num_normal)):
        X_train, X_test, Y_train, Y_test = insert_feature_noise(X, y, num_random_noise=num_normal[i],
            num_overwhelemed=num_overwhelmed[i], num_shortcut=num_shortcut[i])
    
        X_train, X_test, Y_train, Y_test = gini_filter(torch.tensor(X_train.values), torch.tensor(X_test.values), 
            torch.tensor(Y_train, dtype=torch.int64), 
            torch.tensor(Y_test, dtype=torch.int64), 
            left=0.5)

        X_train, X_test, Y_train, Y_test = X_train.numpy(), X_test.numpy(), Y_train.numpy(), Y_test.numpy()
        X_train, Y_train = X_train.astype(np.float32), Y_train.astype(np.int64)
        X_test, Y_test = X_test.astype(np.float32), Y_test.astype(np.int64)

        clf = skw.ScikitClfWrapper(
        input_dim=X_train.shape[1],
        number_of_classes=3,
        hidden_dims=(16, 16), lam=0.1, epochs=1000, sigma=0.5,
        freeze_till=0, lr=0.1, verbose=False,
        device='cpu')
        
        clf.fit(X_train, Y_train)
        acc = print_accuracy(clf.predict, X_test, Y_test)
        two_step_results["accuracy"].append(acc)
        total_feature = 2*num_normal[i] + num_overwhelmed[i] + num_shortcut[i]
        two_step_results['# features'].append(total_feature)
        _, num_gates = clf.model.get_gates()
        print(f"{acc}%, {num_gates}/{total_feature+4} features")




sns.lineplot(data=no_stg_results, x="# features", y="accuracy",
    label='No Feature Selection', marker='o')
sns.lineplot(data=gini_half_results, x="# features", y="accuracy",
    label='Gini Half', marker='o')
sns.lineplot(data=with_stg_results, x="# features", y="accuracy",
    label='STG', marker='o')
sns.lineplot(data=two_step_results, x="# features", y="accuracy",
    label='STG + Gini', marker='o')
plt.show()