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
from re import I
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse
import pandas as pd
from sklearn.model_selection import train_test_split
from FeaturePoison import insert_feature_noise
from Gini import gini_score_fast_old
import torch



# Modify This Class to Support More Dataset
# Alternatively, One can also input numpy dataset 
class VFLDataset(Dataset):
    def __init__(self, filename=None, data_source=None,
                    scale=True, num_clients=1, 
                    feat_idxs=None,
                    gini_portion=None, 
                    insert_noise=False, 
                    num_random_samples=10,
                    num_overwhelemd=5,
                    num_shortcut=5, 
                    noise_std=0.1,
                    noise_skewness=3,
                    noise_type="both",
                    noise_lambda_range=(0, 5), 
                    coefficient_range=(-10, 10),
                    p=0.3, test_size=0.2,
                    permute=True, 
                    seed=0):
        self.permute = permute
        self.insert_noise = insert_noise
        if insert_noise:
            self.num_random_samples = num_random_samples
            self.num_overwhelemd = num_overwhelemd
            self.num_shortcut = num_shortcut
        self.gini_portion = gini_portion
        self.num_clients = num_clients
        self.supported_datasets = {
            'BASEHOCK': "Data/BASEHOCK.mat",
            "PCMAC": "Data/PCMAC.mat"
        }
        if filename in list(self.supported_datasets.keys()):
            print(f'Trying to load the datasets from {filename}')
            filename = self.supported_datasets[filename]
            data_X, data_y = self._load(filename, scale)
        # Else Requiring Numpy Dataset
        # Data source: (X:numpy array, y:numpy array)
        elif filename is None and data_source is not None:
            data_X = data_source[0]
            data_y = data_source[1]
            data_X = data_X.astype(np.float32)
            data_y = data_y.astype(np.int64)
        if insert_noise:
            print(f'Inserting : {num_random_samples} Random Samples, {num_overwhelemd} Overwhelmed Samples, {num_shortcut} Shortcut Samples')
            data_X = pd.DataFrame(data_X)
            # print(data_X)
            # print(data_y)
            X_train, X_test, y_train, y_test = insert_feature_noise(
                data_X, data_y, 
                num_random_noise=num_random_samples, 
                num_overwhelemed=num_overwhelemd, 
                num_shortcut=num_shortcut,
                noise_std=noise_std,
                noise_skewness=noise_skewness,
                noise_type=noise_type, 
                noise_lambda_range=noise_lambda_range,
                coefficient_range=coefficient_range, p=p, 
                test_size=test_size, 
                seed=seed)
            X_train, X_test = X_train.values, X_test.values

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data_X, data_y, test_size=test_size, random_state=seed)

        self.X_train = X_train
        self.X_test = X_test
        # print(X_train.shape)
        # print(X_test.shape)
        self.y_train = y_train
        self.y_test = y_test

        if self.permute:
            self._permute_idx()
    

        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2,
            random_state=seed)
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.X_val = X_val.astype(np.float32)
        self.y_train = y_train.astype(np.int64)
        self.y_test = y_test.astype(np.int64)
        self.y_val = y_val.astype(np.int64)


        # Distribute Dataset to multiple clients
        # data_X, data_y, feat_idx_list = self._load_and_split(filename, 
        # scale, num_clients, feat_idxs)
        feat_idxs_list = self._split(self.X_train, 
            self.y_train, num_clients, feat_idxs)
        self.feat_idxs_list = feat_idxs_list
        self.input_dim_list = [len(idx) for idx in feat_idxs_list]
        
        self.training = 'train'
        
    

    def gini_filter(self, gini_portion):
        X_train = torch.tensor(self.X_train)
        # X_test = torch.tensor(self.X_test), 
        y_train =  torch.tensor(self.y_train, dtype=torch.int64)
        # print(y_train)
        # y_test =  torch.tensor(self.y_test, dtype=torch.int64), 
        gini_score = gini_score_fast_old(X_train, y_train)

        indices = torch.argsort(gini_score)
        # print(indices)
        gini_label = np.zeros(indices.shape[0])
        if isinstance(gini_portion, float):
            indices_left = indices[:int(indices.shape[0]*gini_portion)]
        else: 
            indices_left = indices[:gini_portion]
            
        gini_label[indices_left] = 1
        self.gini_label = gini_label
        return gini_label


    def get_feature_index_list(self):
        return self.feat_idxs_list

    def get_input_dim_list(self):
        return self.input_dim_list

    def _permute_idx(self):
        assert self.X_train.shape[1] == self.X_test.shape[1]
        idx = np.arange(self.X_train.shape[1])
        p = np.random.permutation(idx)
        if self.insert_noise:
            shortcut_label = np.ones(self.X_train.shape[1])
            if self.num_shortcut > 0:
                shortcut_label[p[:self.num_shortcut]] = 0
            overwhelmed_label = np.ones(self.X_train.shape[1])
            if self.num_overwhelemd > 0 and self.num_shortcut > 0:
                overwhelmed_label[-self.num_shortcut-self.num_overwhelemd:-self.num_shortcut] = 0

            random_noise_label = np.ones(self.X_train.shape[1])
            if self.num_shortcut > 0 and self.num_overwhelemd > 0 and self.num_random_samples > 0:
                random_noise_label[-self.num_shortcut-self.num_overwhelemd-self.num_random_samples:-self.num_shortcut-self.num_overwhelemd] = 1
            self.shorcut_label = shortcut_label[p]
            self.overwhelmed_label = overwhelmed_label[p]
            self.random_noise_label = random_noise_label[p]
        self.X_train, self.X_test = self.X_train[:, p], self.X_test[:, p]


    def get_inserted_features_label(self):
        if self.insert_noise:
            return self.random_noise_label, self.shorcut_label, self.overwhelmed_label, 

    def test(self):
        self.training = 'test'
        return self
    
    def train(self):
        self.training = 'train'
        return self
    
    def valid(self):
        self.training = 'valid'
        return self

    def _load(self, filename, scale):   
        try:
            data_mat = loadmat(filename)
            data_X = data_mat['X']
            data_y = data_mat['Y']
            if issparse(data_X):
                data_X = data_X.todense()
            data_y = data_y.flatten()
            print(data_X.shape, data_y.shape)
            data_y[np.where(data_y==1)] = 0
            data_y[np.where(data_y==2)] = 1
            if scale:
                scaler = MinMaxScaler()
                data_X = scaler.fit_transform(data_X)
            return data_X, data_y
        except FileNotFoundError:
            return None

    def _split(self, X, y, num_clients, feat_idxs):
        if feat_idxs is None:
            feat_dim = X.shape[1]
            feat_idxs_list = np.array_split(
                np.arange(feat_dim), num_clients+1)
            for i, feat_idx in enumerate(feat_idxs_list[:-1]):
                print(f'Client {i}: Feature Index {feat_idx[0]}-{feat_idx[-1]}')
            server_idx = feat_idxs_list[-1]
            print(f'Server : Feature Index {server_idx[0]}-{server_idx[-1]}')
        else:
            feat_idxs_list = []
            start = 0
            assert len(feat_idxs) == num_clients and feat_idxs[-1] < feat_dim
            for i, split in enumerate(feat_idxs):
                    feat_idxs_list.append(
                        np.arange(feat_dim)[start: split]
                    )
                    print(f'Client {i}: Feature Index {start}-{split}')
                    start = split
            feat_idxs_list.append(np.arange(feat_dim)[start:])
            print(f'Server : Feature Index {start}-{feat_dim}')
        assert len(feat_idxs_list) == num_clients+1
        return feat_idxs_list
    

    def __len__(self):
        if self.training == 'train':
            return self.X_train.shape[0]
        elif self.training == 'test':
            return self.X_test.shape[0]
        elif self.training == 'valid':
            return self.X_val.shape[0]

    def __getitem__(self, idx):
        if self.training == 'train':
            X = self.X_train
            y = self.y_train
        elif self.training == 'test':
            X = self.X_test
            y = self.y_test
        elif self.training == 'valid':
            X = self.X_val
            y = self.y_val
        items = []
        for i in range(self.num_clients+1):
            item = (X[idx, self.feat_idxs_list[i]])
            items.append(item)
        return items, y[idx]
    

"""
def prepare_data(
    filename, scale=True, num_clients=1, feat_idx=None, 
    train_batch=128, eval_batch=1000, shuffle=False
):
    dataset = VFLDataset(filename, scale, num_clients, feat_idx)
    input_dim_list = dataset.get_input_dim_list()
    train_loader = DataLoader(dataset, batch_size=train_batch, shuffle=shuffle)
    test_loader = DataLoader(dataset.test(), batch_size=eval_batch, shuffle=False)
    return train_loader, test_loader, input_dim_list
"""


if __name__ == "__main__":
    print('Test Dataset Class')
    dataset = VFLDataset(
        "BASEHOCK", scale=True, num_clients=3, feat_idxs=None
    )
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=False)
    val_loader = DataLoader(dataset.valid(), batch_size=1000, shuffle=False)
    print(len(train_loader))
    print(len(test_loader))
    print(len(val_loader))
    dataset = VFLDataset(
        "BASEHOCK", scale=True, num_clients=3, feat_idxs=None,
        insert_noise=True
    )
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=False)
    random_label, over_label, shortcut_label = dataset.get_inserted_features_label()
    print(next(iter(train_loader))[0][0].shape)
    print(next(iter(test_loader))[0][0].shape)
    print(np.count_nonzero(random_label))
    print(np.count_nonzero(over_label))
    print(np.count_nonzero(shortcut_label))
    dataset = VFLDataset(
        data_source=(np.random.randn(1000, 500), np.concatenate((np.ones(500), np.zeros(500)))),
        scale=True, num_clients=3, feat_idxs=None, insert_noise=True
    )
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=False)
    random_label, over_label, shortcut_label = dataset.get_inserted_features_label()
    print(next(iter(train_loader))[0][0].shape)
    print(next(iter(test_loader))[0][0].shape)
    print(np.count_nonzero(random_label))
    print(np.count_nonzero(over_label))
    print(np.count_nonzero(shortcut_label))