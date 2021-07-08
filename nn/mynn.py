import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# A neural network with u_n， θ_c
# u_d = K(θ_c) u_n
# u_d(x) = \int K(x, y, θ_c) u_n(y) dy
class DirectNet_20(nn.Module):

    def __init__(self, N_in, N_out):
        super(DirectNet_20, self).__init__()
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(N_in, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, N_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DirectNet_50(nn.Module):

    def __init__(self, N_in, N_out):
        super(DirectNet_50, self).__init__()
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(N_in, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, N_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# preprocess the training data 

class DirectData(Dataset):

    def __init__(self, X, y):
        
        self.X = X if torch.is_tensor(X) else torch.from_numpy(X)
        self.y = y if torch.is_tensor(y) else torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_bases(K, acc=0.9999, N_trunc=-1):

    N_x, N_y, N_data = K.shape

    data = K.reshape((-1, N_data))

    # svd bases
    u, s, vh = np.linalg.svd(np.transpose(data))
    
    if N_trunc < 0:
        s_sum_tot = np.dot(s, s)
        s_sum = 0.0
        for i in range(N_data):
            s_sum += s[i]**2
            if s_sum > acc*s_sum_tot:
                break
        N_trunc = i+1
    print("N_trunc = ", N_trunc)



    scale = np.average(s[:N_trunc])
    data_svd = u[:, 0:N_trunc] * s[:N_trunc]/scale
    bases = vh[0:N_trunc, :]*scale

    return data_svd, bases, N_trunc, s


def full2upper(K):
    N_x, N_y = K.shape
    upper = np.zeros(N_x *(N_y + 1) // 2)
    i = 0
    for i_x in range(N_x):
        for i_y in range(i_x+1):
            upper[i] = K[i_x, i_y]
            i += 1
    return upper


def upper2full(upper):
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