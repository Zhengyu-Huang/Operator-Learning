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

s = N = 200
ntrain = M//2
ntest = M-M//2

prefix = "../../../data/"  
a0 = np.load(prefix+"adv_a0.npy")
aT = np.load(prefix+"adv_aT.npy")

inputs  = np.copy(a0)
outputs = np.copy(aT)

# transpose
a0 = a0.transpose(1, 0)
aT = aT.transpose(1, 0)


xgrid = np.linspace(0,1,N)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



x_train = torch.from_numpy(np.reshape(a0[:M//2,   :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(aT[:M//2,   :], -1).astype(np.float32))

x_test = torch.from_numpy(np.reshape(a0[M//2:M,   :], -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(aT[M//2:M,   :], -1).astype(np.float32))


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)






x_train = x_train.reshape(ntrain,s, 1)
x_test = x_test.reshape(ntest,s, 1)

# todo do we need this
y_train = y_train.reshape(ntrain,s, 1)
y_test = y_test.reshape(ntest,s,1)


model = torch.load("FNO_"+str(width)+"Nd_"+str(ntrain)+".model", map_location=device)

# Training error
rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    print("i / N = ", i, " / ", M//2)
    aT_train = y_normalizer.decode(model(x_train[i:i+1, :, :].to(device))).detach().cpu().numpy()
    rel_err_nn_train[i] =  np.linalg.norm(aT_train - y_normalizer.decode(y_train[i, :]).cpu().numpy())/np.linalg.norm(y_normalizer.decode(y_train[i, :]).cpu().numpy())
mre_nn_train = np.mean(rel_err_nn_train)

########### Test
rel_err_nn_test = np.zeros(M-M//2)
for i in range(M-M//2):
    print("i / N = ", i, " / ", M-M//2)
    aT_test = y_normalizer.decode(model(x_test[i:i+1, :].to(device))).detach().cpu().numpy()
    rel_err_nn_test[i] =  np.linalg.norm(aT_test - y_test[i, :].cpu().numpy())/np.linalg.norm(y_test[i, :].cpu().numpy())
mre_nn_test = np.mean(rel_err_nn_test)

print("NN: rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)



# save smallest, medium, largest
test_input_save = np.zeros((N, 3))
test_output_save = np.zeros((N, 6))
for i, ind in enumerate([np.argmin(rel_err_nn_test), np.argsort(rel_err_nn_test)[len(rel_err_nn_test)//2], np.argmax(rel_err_nn_test)]):
    test_input_save[:, i] = inputs[:, M//2 + ind]
    # truth
    test_output_save[:, i] = outputs[:, M//2 + ind]
    # predict
    test_output_save[:, i + 3] = y_normalizer.decode(model(x_test[ind:ind+1, :].to(device))).detach().cpu().numpy().flatten()
    
    
np.save(str(ntrain) + "_" + str(width) + "_test_input_save.npy", test_input_save)
np.save(str(ntrain) + "_" + str(width) + "_test_output_save.npy", test_output_save)
