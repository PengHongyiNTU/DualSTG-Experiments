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



from os import access

from sklearn.metrics import accuracy_score
from Model import STGEmbModel, FNNModel
import torch
import numpy as np
import pandas as pd



class FNNClfWrapper:
    def __init__(self, input_dim, number_of_classes, hidden_dims,
        batch_norm=None, dropout=None, activation='relu',
        flatten=True, epochs=10, verbose=False, device='cpu', lr=0.001):
        self.model = FNNModel(input_dim, number_of_classes, hidden_dims,
            batch_norm=batch_norm, dropout=dropout, activation=activation,
            flatten=flatten)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.epochs = epochs
        self.verbose = verbose
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def fit(self, X, y):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        for e in range(self.epochs):
            train_loss = 0
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            if self.verbose:
                print(f'Epoch {e+1}/{self.epochs} Loss: {train_loss/len(train_loader)}')

    def predict(self, x):
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32))
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for x in test_loader:
            x = x[0].to(self.device)
            y_pred = self.model(x)
            y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
            return y_pred.argmax(dim=1).cpu().numpy()




class ScikitClfWrapper:
    def __init__(self,  input_dim, number_of_classes, 
                 hidden_dims, sigma=1.0, lam=0.1,
                 batch_norm=None, dropout=None, activation='relu',
                 flatten=True, 
                 epochs=500, lr=1e-3, 
                 device='cpu', verbose=False,
                 freeze_till=0, 
                 early_stopping=False, patience=50, valid_split=0.7):

        self.model = STGEmbModel(
            input_dim=input_dim, output_dim=number_of_classes, activation=activation,
            hidden_dims=hidden_dims, sigma=sigma, lam=lam,
            batch_norm=batch_norm, dropout=dropout, flatten=flatten)
        self.epochs = epochs
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device(device)
        self.verbose = verbose
        self.freeze_till = freeze_till
        self.model.to(device)
        self.early_stopping = early_stopping
        self.patience = patience
        self.valid_split = valid_split
        self.current_best = 0
        self.count = 0
        self.history = []
        
    def get_history(self):
        history = pd.DataFrame(self.history, columns=['epoch', 'valid_acc', '# feature'])
        return history

    def _accuracy(self, y_pred, y_true):
        return 100*np.sum(y_pred == y_true)/len(y_true)

    def fit(self, X, y):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))
        split = int(len(dataset)*self.valid_split)
        train_set, val_set = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)
        for e in range(self.epochs):
            if e >= self.freeze_till:
                self.model.unfreeze_fs()
            else:
                self.model.freeze_fs()
            train_loss = 0
            for x, y in train_loader:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                # loss = criterion(outputs, labels)
                loss = self.criterion(y_pred, y)
                reg_loss = self.model.get_reg_loss()
                total_loss = loss + reg_loss
                total_loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, num_feat = self.model.get_gates()
            for x, y in val_loader:
                with torch.no_grad():
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y_pred = self.model(x)
                    accuracy_score = self._accuracy(y_pred.argmax(dim=1).cpu().numpy(), y.cpu().numpy())
            
            if self.verbose:
                print(f'Epoch {e+1}/{self.epochs} Loss: {train_loss/len(train_loader)} num_feat: {num_feat}/{x.shape[1]} Reg Loss: {reg_loss} Val Accuracy: {accuracy_score}')
            
            self.history.append([e, accuracy_score, num_feat])
            
            if self.early_stopping:
                if self.current_best < accuracy_score:
                    self.current_best = accuracy_score
                    self.count = 0
                else:
                    self.count += 1
                if self.count >= self.patience:
                    print("Early Stopped at {} at {}".format(self.current_best, e))
                    break
            


    def predict(self, X):
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for x in test_loader:
            x = x[0].to(self.device)
            y_pred = self.model(x)
            y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
            return y_pred.argmax(dim=1).cpu().numpy()
