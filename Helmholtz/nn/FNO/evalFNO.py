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

M = int(sys.argv[1]) #5000
width = int(sys.argv[2])
batch_size = int(sys.argv[3])

N = 100
ntrain = M//2
ntest = M-M//2
s = N+1

N_theta = 100
prefix = "../../../data/"  
K = np.load(prefix+"Random_Helmholtz_high_K_" + str(N_theta) + ".npy")
cs = np.load(prefix+"Random_Helmholtz_high_cs_" + str(N_theta) + ".npy")

inputs = np.copy(cs)
outputs = np.copy(K)

xgrid = np.linspace(0,1,N+1)
dx    = xgrid[1] - xgrid[0]
Y, X = np.meshgrid(xgrid, xgrid)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12



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
vmin, vmax = y_normalizer.decode(y_train[i,:,:,0]).detach().cpu().numpy().min(), y_normalizer.decode(y_train[i,:,:,0]).detach().cpu().numpy().max()
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
vmin, vmax = y_normalizer.decode(y_test[i,:,:,0]).detach().cpu().numpy().min(), y_normalizer.decode(y_test[i,:,:,0]).detach().cpu().numpy().max()
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


#########################################
# save smallest, medium, largest
test_input_save  = np.zeros((N+1,  N+1, 3))
test_output_save = np.zeros((N+1,  N+1, 6))
for i, ind in enumerate([np.argmin(rel_err_nn_test), np.argsort(rel_err_nn_test)[len(rel_err_nn_test)//2], np.argmax(rel_err_nn_test)]):
    test_input_save[:, :, i] = inputs[:, :, M//2 + ind]
    # truth
    test_output_save[:, :, i] = outputs[:, :, M//2 + ind]
    # predict
    test_output_save[:, :, i + 3] =  y_normalizer.decode(model(x_test[ind:ind+1, :, :, :].to(device))).detach().cpu().numpy()[0,:,:,0]

np.save(str(ntrain) + "_" + str(width) + "_test_input_save.npy", test_input_save)
np.save(str(ntrain) + "_" + str(width) + "_test_output_save.npy", test_output_save)



