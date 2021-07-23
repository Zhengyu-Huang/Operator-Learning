import numpy as np
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


