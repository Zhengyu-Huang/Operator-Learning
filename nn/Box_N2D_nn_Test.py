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
        
        self.fc1 = nn.Linear(N_θ + 2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



test_data = True

prefix = "test_" if test_data else "" 


L = 1.0
# load data 
θ = np.load(prefix + "uniform_direct_theta.npy")
κ = np.load(prefix + "uniform_direct_K.npy")

N_data, N_θ =  θ.shape
N_x, N_y, N_data = κ.shape

assert(N_x == N_y)
Δx = L/(N_x - 1)


# load the model
model = torch.load("DirectKernelNet.model")
   


xx = np.linspace(0, L, N_x)
Y, X = np.meshgrid(xx, xx)


test_id = 1
input_test = np.zeros((N_x * N_y, (N_θ + 2)), dtype=np.float32) # θ, x, y
output_test = np.zeros((N_x * N_y), dtype=np.float32)

input_test[: , 0:N_θ] = θ[test_id]
input_test[: , N_θ] = X.reshape(-1)
input_test[: , N_θ+1] = Y.reshape(-1)
output_test = model(torch.from_numpy(input_test))
κ_pred = output_test.detach().numpy().reshape((N_x, N_y))

vmin, vmax = np.min(κ[:, :, test_id]), np.max(κ[:, :, test_id])
fig = plt.figure()
plt.pcolormesh(X, Y, κ[:, :, test_id], shading="gouraud", vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title("Truth")
fig.savefig(prefix+"Truth%i.png" % test_id)

fig = plt.figure()
plt.pcolormesh(X, Y, κ_pred, shading="gouraud", vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title("Prediction")
fig.savefig(prefix+"Prediction%i.png" % test_id)

print(prefix+"data %i, relative error is %.5f:" % (test_id, np.linalg.norm(κ_pred - κ[:, :, test_id])/np.linalg.norm(κ[:, :, test_id])))

