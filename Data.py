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
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import issparse




# Modify This Class to Support More Dataset

class VFLDataset(Dataset):
    def __init__(self, filename, scale=True, num_clients=1, feat_idxs=None):
        self.num_clients = num_clients
        print(f'Trying to load the datasets from {filename}')
        if filename == "BASEHOCK":
            filename = "Data/BASEHOCK.mat"
        if filename == "PCMAC":
            filename = "Data/PCMAC.mat"
        data_X, data_y, feat_idx_list = self._load_and_split(filename, 
            scale, num_clients, feat_idxs)
        self.data_X = data_X
        self.data_y = data_y
        self.feat_idxs_list = feat_idx_list
        self.input_dim_list = [len(idx) for idx in feat_idx_list]
    
    def get_input_dim_list(self):
        return self.input_dim_list

    def _load_and_split(self, filename, scale, num_clients, feat_idxs):   
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
        except FileNotFoundError:
            return None
        if feat_idxs is None:
            feat_dim = data_X.shape[1]
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
        return data_X, data_y, feat_idxs_list
    

    def __len__(self):
        return self.data_X.shape[0]

    def __getitem__(self, idx):
        items = []
        for i in range(self.num_clients+1):
            item = (self.data_X[idx, self.feat_idxs_list[i]])
            items.append(item)
        return items, self.data_y[idx]
    

def prepare_data(
    filename, scale=True, num_clients=1, feat_idx=None, 
    train_batch=128, eval_batch=1000, shuffle=False
):
    dataset = VFLDataset(filename, scale, num_clients, feat_idx)
    input_dim_list = dataset.get_input_dim_list()
    train_loader = DataLoader(dataset, batch_size=train_batch, shuffle=shuffle)
    test_loader = DataLoader(dataset, batch_size=eval_batch, shuffle=False)
    return train_loader, test_loader, input_dim_list


if __name__ == "__main__":
    train_loader, test_loader, input_dim_list  = prepare_data("BASEHOCK", scale=True, num_clients=2)
    print(next(iter(train_loader)))