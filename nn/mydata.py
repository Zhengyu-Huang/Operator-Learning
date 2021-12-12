import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def preprocess(params, dim, X, data):
    # params:  M by Np
    # X:  M by Ny by Nz ....
    # data: M by Nx by Ny by Nz ....

    M, _ = params.shape 
    if dim == 1:
        Nx = len(X)
        X_data = np.concatenate((np.repeat(params,Nx,axis=0), np.tile(X,M)[:,np.newaxis]),axis=1)
        y_data = np.reshape(data,(M*Nx, 1))

    
    return X_data, y_data



    # def __initialize(self):
    #     for i in range(1, self.trunk_depth):
    #         self.weight_init_(self.modus['TrLinM{}'.format(i)].weight)
    #         nn.init.constant_(self.modus['TrLinM{}'.format(i)].bias, 0)
# preprocess the training data 

class DirectData(Dataset):

    def __init__(self, X, y):
        
        self.X = X if torch.is_tensor(X) else torch.from_numpy(X)
        self.y = y if torch.is_tensor(y) else torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# A basic implementation, features are put in the front
# Vs is a M by d vectors, it will help generate 2M additional features
#    from the first d features
def EnrichData(X_data, Vs):
    N_data, N_dim = X_data.shape
    M, d = Vs.shape
    N_dim_new = N_dim + 2*M
    X_data_new = np.zeros((N_data, N_dim_new))
    X_data_new[:, -N_dim:] =  X_data
    for i in range(N_data):
        for j in range(M):
            X_data_new[i, 2*j]   = np.sin(np.dot(Vs[j, :], X_data[i, 0:d]))
            X_data_new[i, 2*j+1] = np.cos(np.dot(Vs[j, :], X_data[i, 0:d]))
    return X_data_new
    
def preprocess_PCA(params, data, acc=0.9999, N_trunc=-1):
    # params:  M by Np
    # X:  M by Ny by Nz ....
    # data: M by Nx by Ny by Nz ....

    X_train = params[0::2, :]
    X_test  = params[1::2, :]

    N_data = data.shape[0]

    data = data.reshape((N_data, -1))
    
    # svd bases
    u, s, vh = np.linalg.svd(data[0::2])
    
    if N_trunc < 0:
        s_sum_tot = np.dot(s, s)
        s_sum = 0.0
        for i in range(N_data):
            s_sum += s[i]**2
            if s_sum > acc*s_sum_tot:
                break
        N_trunc = i+1
    


    
    bases = vh[0:N_trunc, :].T

    data_svd = np.dot(data, bases)

    y_train = data_svd[0::2, :]
    y_test = data_svd[1::2, :]

    return bases, X_train, y_train, X_test, y_test


def full2upper(K):
    N_x, N_y = K.shape
    upper = np.zeros(N_x *(N_y + 1) // 2)
    i = 0
    for i_x in range(N_x):
        for i_y in range(i_x+1):
            upper[i] = K[i_x, i_y]
            i += 1
    return upper


def upper2full_1(upper):
    N_x = N_y = int((np.sqrt(8*len(upper) + 1) - 1)/2)
    K = np.zeros((N_x , N_y))
    i = 0
    for i_x in range(N_x):
        for i_y in range(i_x+1):
            K[i_x, i_y] = K[i_y, i_x]  = upper[i]
            i += 1
    return K

def upper2full(upper, N_data):
    N_x = N_y = int((np.sqrt(8*len(upper)/N_data + 1) - 1)/2)
    N_upper =  N_x *(N_y + 1) //2
    K = np.zeros((N_x , N_y, N_data))

    for i_data in range(N_data):
        i = 0
        for i_x in range(N_x):
            for i_y in range(i_x+1):
                K[i_x, i_y, i_data] = K[i_y, i_x, i_data]  = upper[i + N_upper*i_data]
                i += 1
    return K


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        # x = (x - self.mean) / (self.std + self.eps)
        # return x

        # x -= self.mean
        # x /= (self.std + self.eps)
        return (x - self.mean) / (self.std + self.eps)
    
    def encode_(self, x):
        # x = (x - self.mean) / (self.std + self.eps)
        # return x

        x -= self.mean
        x /= (self.std + self.eps)
        

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x * std) + mean
        # return x

        # x *= std 
        # x += mean
        return (x * std) + mean

    def decode_(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x * std) + mean
        # return x

        x *= std 
        x += mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()