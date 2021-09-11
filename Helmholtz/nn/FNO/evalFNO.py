"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import sys
sys.path.append('../../../nn')
from mynn import *
from mydata import UnitGaussianNormalizer
from Adam import Adam

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from datetime import datetime

import matplotlib as mpl 
from matplotlib.lines import Line2D 
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
plt.rc("text", usetex=True)         # Crisp axis ticks
plt.rc("font", family="serif")      # Crisp axis labels
plt.rc("legend", edgecolor='none')  # No boxes around legends

plt.rc("figure",facecolor="#ffffff")
plt.rc("axes",facecolor="#ffffff",edgecolor="#000000",labelcolor="#000000")
plt.rc("savefig",facecolor="#ffffff")
plt.rc("text",color="#000000")
plt.rc("xtick",color="#000000")
plt.rc("ytick",color="#000000")

color1 = 'tab:blue'
color2 = 'tab:green'
color3 = 'tab:orange'

import operator
from functools import reduce
from functools import partial

from timeit import default_timer


torch.manual_seed(0)
np.random.seed(0)

N = 100
M = 5000
ntrain = M//2
ntest = M-M//2
s = N+1

N_theta = 100
prefix = "../"  
K = np.load(prefix+"Random_Helmholtz_K_" + str(N_theta) + ".npy")
cs = np.load(prefix+"Random_Helmholtz_cs_" + str(N_theta) + ".npy")


xgrid = np.linspace(0,1,N+1)
dx    = xgrid[1] - xgrid[0]
Y, X = np.meshgrid(xgrid, xgrid)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
# TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
# TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'



batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12
width = 32



################################################################
# load data and data normalization
################################################################


# transpose
cs = cs.transpose(2, 0, 1)
K = K.transpose(2, 0, 1)

x_train = torch.from_numpy(np.reshape(cs[:M//2, :, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(K[:M//2, :, :], -1).astype(np.float32))

x_test = torch.from_numpy(np.reshape(cs[M//2:M, :, :], -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(K[M//2:M, :, :], -1).astype(np.float32))


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)






x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)

# todo do we need this
y_train = y_train.reshape(ntrain,s,s,1)
y_test = y_test.reshape(ntest,s,s,1)


model = torch.load("FNO_"+str(width)+"Nd_"+str(ntrain)+".model", map_location=device)

# Training error
rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    print("i / N = ", i, " / ", M//2)
    K_train = y_normalizer.decode(model(x_train[i:i+1, :, :, :].to(device))).detach().cpu().numpy()
    rel_err_nn_train[i] =  np.linalg.norm(K_train - y_normalizer.decode(y_train[i, :, :]).cpu().numpy())/np.linalg.norm(y_normalizer.decode(y_train[i, :, :]).cpu().numpy())
mre_nn_train = np.mean(rel_err_nn_train)

####### worst error plot
i = np.argmax(rel_err_nn_train)
K_train = y_normalizer.decode(model(x_train[i:i+1, :, :, :].to(device))).detach().cpu().numpy()
fig,ax = plt.subplots(ncols=3, figsize=(9,3))
vmin, vmax = y_normalizer.decode(y_train[i,:,:]).detach().cpu().numpy().min(), y_normalizer.decode(y_train[i,:,:]).detach().cpu().numpy().max()
ax[0].pcolormesh(X, Y, x_normalizer.decode(x_train[i, :, :, 0]).detach().cpu().numpy(),  shading='gouraud')
ax[1].pcolormesh(X, Y, K_train[0,:,:,0], shading='gouraud', vmin=vmin, vmax =vmax)
ax[2].pcolormesh(X, Y, y_normalizer.decode(y_train[i,:,:, 0]).detach().cpu().numpy(), shading='gouraud', vmin=vmin, vmax =vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('worst_case_train_NN.png', pad_inches=3)
plt.close()


########### Test
rel_err_nn_test = np.zeros(M-M//2)
for i in range(M-M//2):
    print("i / N = ", i, " / ", M-M//2)
    K_test = y_normalizer.decode(model(x_test[i:i+1, :, :, :].to(device))).detach().cpu().numpy()
    rel_err_nn_test[i] =  np.linalg.norm(K_test - y_test[i, :, :].cpu().numpy())/np.linalg.norm(y_test[i, :, :].cpu().numpy())
mre_nn_test = np.mean(rel_err_nn_test)



####### worst error plot
i = np.argmax(rel_err_nn_test)
K_test = y_normalizer.decode(model(x_test[i:i+1, :, :, :].to(device))).detach().cpu().numpy()
fig,ax = plt.subplots(ncols=3, figsize=(9,3))
vmin, vmax = y_normalizer.decode(y_test[i,:,:]).detach().cpu().numpy().min(), y_normalizer.decode(y_test[i,:,:]).detach().cpu().numpy().max()
ax[0].pcolormesh(X, Y, x_normalizer.decode(x_test[i, :, :, 0]).detach().cpu().numpy(),  shading='gouraud')
ax[1].pcolormesh(X, Y, K_test[0,:,:,0], shading='gouraud', vmin=vmin, vmax =vmax)
ax[2].pcolormesh(X, Y, y_test[i,:,:, 0].detach().cpu().numpy(), shading='gouraud', vmin=vmin, vmax =vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('worst_case_test_NN.png', pad_inches=3)
plt.close()



fig,ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(rel_err_nn_train,lw=0.5,color=color1,label='training')
ax.semilogy(rel_err_nn_test,lw=0.5,color=color2,label='test')
ax.legend()
plt.xlabel('data index')
plt.ylabel('Relative errors')
plt.tight_layout()
plt.savefig('NN_errors.png',pad_inches=3)
plt.close()

print("NN: rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)



