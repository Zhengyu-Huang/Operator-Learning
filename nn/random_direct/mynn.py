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
class DirectKernelNet(nn.Module):

    def __init__(self, N_θ):
        super(DirectKernelNet, self).__init__()
        self.N_θ = N_θ
        # an affine operation: y = Wx + b
        
        self.fc1 = nn.Linear(N_θ + 2, 20)
        self.fc2 = nn.Linear(20, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 20)
        self.fc5 = nn.Linear(20, 1)

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

def preprocess_data(seeds = []):
    # concatenate data
    θs, κs = [], []

    if not seeds:
        θ = np.load("random_direct_theta.npy")   
        κ = np.load("random_direct_K.npy")
    else:
        # load data 
        for seed in seeds:
            print("load random_direct_theta."+str(seed)+".npy and random_direct_K."+str(seed)+".npy")
            θs.append(np.load("random_direct_theta."+str(seed)+".npy"))
            κs.append(np.load("random_direct_K."+str(seed)+".npy"))

        θ = np.concatenate(θs, axis = 0)   
        κ = np.concatenate(κs, axis = 2)


    N_data, N_θ =  θ.shape
    N_x, N_y, N_data = κ.shape


    input_train  = np.zeros((N_data * N_x * N_y, (N_θ + 2)), dtype=np.float32) # θ, x, y
    output_train = np.zeros((N_data * N_x * N_y), dtype=np.float32)


    L = 1.0
    assert(N_x == N_y)
    Δx = L/(N_x - 1)
    xx = np.linspace(0, L, N_x)
    Y, X = np.meshgrid(xx, xx)

    # test
    i = 20
    j = 40
    assert(X[i, j] == i*Δx and Y[i, j] == j*Δx)



    for i in range(N_data):
        d_range = range(i*N_x*N_y, (i + 1)*N_x*N_y)
        input_train[d_range , 0:N_θ] = θ[i]
        input_train[d_range , N_θ] = X.reshape(-1)
        input_train[d_range , N_θ + 1] = Y.reshape(-1)
        output_train[d_range] = κ[:, :, i].reshape(-1)
    

    # input out put data only choose x <= y
    input_bool = (input_train[:, N_θ] <= input_train[:, N_θ+1] + Δx/2)
    input_train = input_train[input_bool, :]
    output_train = output_train[input_bool]

    return input_train, output_train