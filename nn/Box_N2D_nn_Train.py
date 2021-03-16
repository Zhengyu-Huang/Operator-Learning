import numpy as np
import matplotlib.pyplot as plt
import math
import py
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


# preprocess the training data 

class DirectData(Dataset):

    def __init__(self, X, y):
        
        self.X = X if torch.is_tensor(X) else torch.from_numpy(X)
        self.y = y if torch.is_tensor(y) else torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


L = 1.0
# load data 
θ = np.load("uniform_direct_theta.npy")
κ = np.load("uniform_direct_K.npy")

N_data, N_θ =  θ.shape
N_x, N_y, N_data = κ.shape

assert(N_x == N_y)
Δx = L/(N_x - 1)

# todo set number of training data
N_data = 1

input_train  = np.zeros((N_data * N_x * N_y, (N_θ + 2)), dtype=np.float32) # θ, x, y
output_train = np.zeros((N_data * N_x * N_y), dtype=np.float32)

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
    



x_train = torch.from_numpy(input_train) 
y_train = torch.from_numpy(output_train).unsqueeze(-1)


ds = DirectData(X=x_train, y=y_train)
ds = DataLoader(ds, batch_size=512, shuffle=True)



# training with adam
model = DirectKernelNet(N_θ)
loss_fn = torch.nn.MSELoss(reduction='sum')


learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

n_epochs = 50000
for epoch in range(n_epochs):
    
    for ix, (_x, _y) in enumerate(ds):
    
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(_x)

        # Compute and print loss.
        loss = loss_fn(y_pred, _y)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
    if epoch % 100 == 0:
        print("[{}/{}], loss: {}".format(epoch, n_epochs, np.round(loss.item(), 3)))


# save the model
torch.save(model, "DirectKernelNet.model")

# test on training data
test_id = 0

input_test = np.zeros((N_x * N_y, (N_θ + 2)), dtype=np.float32) # θ, x, y
output_test = np.zeros((N_x * N_y), dtype=np.float32)

input_test[: , 0:N_θ] = θ[test_id]
input_test[: , N_θ] = X.reshape(-1)
input_test[: , N_θ+1] = Y.reshape(-1)
output_test = model(torch.from_numpy(input_test))
κ_pred = output_test.detach().numpy().reshape((N_x, N_y))

vmin, vmax = np.min(κ[:, :, test_id]), np.max(κ[:, :, test_id])
fig = plt.figure()
plt.pcolormesh(X, Y, κ[:, :, test_id], shading="gouraud")
plt.colorbar()
plt.title("Truth")
fig.savefig("Truth%i.png" % test_id)

fig = plt.figure()
plt.pcolormesh(X, Y, κ_pred, shading="gouraud")
plt.colorbar()
plt.title("Prediction")
fig.savefig("Prediction%i.png" % test_id)

