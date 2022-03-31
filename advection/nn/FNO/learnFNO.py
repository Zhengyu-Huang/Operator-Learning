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

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer



# torch.manual_seed(0)
# np.random.seed(0)






################################################################
# load data and data normalization
################################################################
print(sys.argv)


M = int(sys.argv[1]) #5000
width = int(sys.argv[2])
batch_size = int(sys.argv[3])






s = N = 200
ntrain = M//2
ntest = M//2
N_theta = 100
prefix = "../../../data/"  
a0 = np.load(prefix+"adv_a0.npy")
aT = np.load(prefix+"adv_aT.npy")



# transpose
a0 = a0.transpose(1, 0)
aT = aT.transpose(1, 0)

x_train = torch.from_numpy(np.reshape(a0[:M//2, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(aT[:M//2, :], -1).astype(np.float32))

x_test = torch.from_numpy(np.reshape(a0[M//2:M,   :], -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(aT[M//2:M,   :], -1).astype(np.float32))


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)


x_train = x_train.reshape(ntrain,s ,1)
x_test = x_test.reshape(ntest,s ,1)

# todo do we need this
y_train = y_train.reshape(ntrain ,s,1)
y_test = y_test.reshape(ntest ,s,1)



################################################################
# training and evaluation
################################################################

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

modes = 12


model = FNO1d(modes, width).cuda()
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
t0 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        batch_size_ = x.shape[0]
        optimizer.zero_grad()
        out = model(x).reshape(batch_size_,   s)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    torch.save(model, "FNO_"+str(width)+"Nd_"+str(ntrain)+".model")
    scheduler.step()

    train_l2/= ntrain

    t2 = default_timer()
    print("Epoch : ", ep, " Epoch time : ", t2-t1, " Rel. Train L2 Loss : ", train_l2)

print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)
