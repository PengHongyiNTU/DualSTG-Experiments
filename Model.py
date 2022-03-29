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

from pyrsistent import freeze
from stg.models import MLPLayer
from torch import nn 
import torch 
import numpy as np
import math




class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) 
        self.sigma = sigma
        self.freeze = False
    
    def forward(self, prev_x):
        if self.freeze:
            return prev_x
            
        elif self.training:
            z = self.mu + self.sigma*self.noise.normal_()*self.training 
            stochastic_gate = self.hard_sigmoid(z)
            new_x = prev_x * stochastic_gate
            return new_x
        else:
            z = self.hard_sigmoid(self.mu)
            return prev_x * z        

    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self



class FNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims,
        batch_norm=None, dropout=None, activation='relu',
        flatten=True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.mlp = MLPLayer(
            input_dim, output_dim=output_dim, 
            hidden_dims=hidden_dims, batch_norm=batch_norm,
            dropout=dropout, activation=activation,
            flatten=flatten)   
        
    def forward(self, x):
        return self.mlp(x)
        

class STGEmbModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, 
            sigma=1.0, lam=0.1, 
            batch_norm=None, dropout=None, activation='relu',
            flatten=True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.mlp = MLPLayer(input_dim, output_dim, hidden_dims, 
            batch_norm=batch_norm, dropout=dropout, activation=activation,
            flatten=flatten)
        self.fs = FeatureSelector(input_dim, sigma)
        self.reg = self.fs.regularizer
        self.lam = lam
        self.mu = self.fs.mu
        self.sigma = self.fs.sigma
    
    def forward(self, x):
        x = self.fs(x)
        emb = self.mlp(x)
        return emb

    def freeze_fs(self):
        for param in self.fs.parameters():
            param.requires_grad = False
        self.fs.freeze = True

    def unfreeze_fs(self):
        for param in self.fs.parameters():
            param.requires_grad = True 
        self.fs.freeze = False
    
    def get_reg_loss(self):
        reg = torch.mean(self.reg((self.mu + 0.5)/self.sigma)) 
        return reg * self.lam
    
    def get_gates(self):
        mu = self.mu.detach().cpu()
        z = torch.clamp(mu+0.5, 0.0, 1.0).numpy()
        return z, np.count_nonzero(z)
    
class DualSTGModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, 
            btm_sigma=1.0, btm_lam=0.1, top_sigma=1.0, top_lam=0.1, 
            batch_norm=None, dropout=None, activation='relu',
            flatten=True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.mlp = MLPLayer(input_dim, output_dim, hidden_dims, 
            batch_norm=batch_norm, dropout=dropout, activation=activation,
            flatten=flatten)
        self.btm_fs = FeatureSelector(input_dim, btm_sigma)
        self.btm_reg = self.btm_fs.regularizer
        self.btm_lam = btm_lam
        self.btm_mu = self.btm_fs.mu
        self.btm_sigma = self.btm_fs.sigma
        ################################################################
        self.top_fs = FeatureSelector(output_dim, top_sigma)
        self.top_reg = self.top_fs.regularizer
        self.top_lam = top_lam
        self.top_sigma  = self.top_fs.sigma
        self.top_mu = self.top_fs.mu
        ################################################################

    def freeze_top(self):
        for param in self.top_fs.parameters():
            param.requires_grad = False
    
    def unfreeze_top(self):
        for param in self.top_fs.parameters():
            param.requires_grad = True


    def freeze_fs(self):
        for param in self.btm_fs.parameters():
            param.requires_grad = False

    def unfreeze_fs(self):
        for param in self.btm_fs.parameters():
            param.requires_grad = True 

    def forward(self, x):
        x = self.btm_fs(x)
        emb = self.mlp(x)
        reduced_emb = self.top_fs(emb)
        return reduced_emb

    def get_top_reg_loss(self):
        top_reg_loss= torch.mean(
            self.top_reg((self.top_mu + 0.5)/self.top_sigma))
        return top_reg_loss * self.top_lam
    
    def get_btm_reg_loss(self):
        btm_reg_loss = torch.mean(
            self.btm_reg((self.btm_mu + 0.5)/self.btm_sigma))
        return btm_reg_loss * self.btm_lam
    
    def get_reg_loss(self):
        top_reg_loss = self.get_top_reg_loss()
        btm_reg_loss = self.get_btm_reg_loss()
        return top_reg_loss + btm_reg_loss

    def get_gates(self):
        top_mu = self.top_mu.detach().cpu()
        top_z = torch.clamp(top_mu+0.5, 0.0, 1.0).numpy()
        btm_mu = self.btm_mu.detach().cpu()
        btm_z = torch.clamp(btm_mu+0.5, 0.0, 1.0).numpy()
        return top_z, btm_z, np.count_nonzero(top_z), np.count_nonzero(btm_z)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = FNNModel(input_dim=2, output_dim=2, hidden_dims=[10, 10],
        batch_norm=None, dropout=None, activation='relu',
        flatten=True).cpu()
    model = model.to(device)
    print('*'*89)
    for name, parameter in model.named_parameters():
        print(name, parameter.shape)
    model = STGEmbModel(
        input_dim=2, output_dim=2, hidden_dims=[10, 10]
    )
    model = model.to(device)
    print('*'*89)
    for name, parameter in model.named_parameters():
        print(name, parameter.shape)
    model = DualSTGModel(
        input_dim=2, output_dim=2, hidden_dims=[10, 10],
        btm_sigma=1.0, btm_lam=0.1, top_sigma=1.0, top_lam=0.1,
        batch_norm=None, dropout=None, activation='relu',
        flatten=True)
    model = model.to(device)
    print('*'*89)
    for name, parameter in model.named_parameters():
        print(name, parameter.shape)
    

    
