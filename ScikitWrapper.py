
from Model import STGEmbModel, DualSTGModel
import torch


class ScikitClfWrapper:
    def __init__(self,  input_dim, number_of_classes, 
                 hidden_dims, sigma=1.0, lam=0.0,
                 batch_norm=None, dropout=None, activation='relu',
                 flatten=True, 
                 epochs=500, lr=1e-2, 
                 device='cpu', verbose=False,
                 freeze_at=0):
        self.model = STGEmbModel(
            input_dim=input_dim, output_dim=number_of_classes, activation=activation,
            hidden_dims=hidden_dims, sigma=sigma, lam=lam,
            batch_norm=batch_norm, dropout=dropout, flatten=flatten)
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device(device)
        self.verbose = verbose
        self.freeze_at = freeze_at
        self.model.to(device)
        
    def fit(self, X, y):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
        for e in range(self.epochs):
            if e <= self.freeze_at:
                self.model.freeze_fs()
            else:
                self.model.unfreeze_fs()
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
            if self.verbose:
                print(f'Epoch {e+1}/{self.epochs} Loss: {train_loss/len(train_loader)} num_feat: {num_feat} Reg Loss: {reg_loss}')
            
    def predict(self, X):
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for x in test_loader:
            x = x[0].to(self.device)
            y_pred = self.model(x)
            y_pred = torch.nn.functional.log_softmax(y_pred, dim=1)
            return y_pred.argmax(dim=1).numpy()
