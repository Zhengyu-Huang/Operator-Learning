import sys
import numpy as np
sys.path.append('../../../nn')
from mynn import *
from datetime import datetime

import matplotlib as mpl 
from matplotlib.lines import Line2D 
mpl.use('TkAgg')
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


def colnorm(u):
	return np.sqrt(np.sum(u**2,0))

N = 256
K = 200
M = 2048

xgrid = np.linspace(0,1,N+1)
xgrid = xgrid[:-1]
dx    = xgrid[1] - xgrid[0]

# burgers param and data
nu      = 0.01
data    = np.load('../../data/N'+str(N)+'_K'+str(K)+'_M'+str(M)+'.npz')
inputs  = data["inputs"]
outputs = data["outputs"]

train_inputs = inputs[:,:M/2]
test_inputs  = inputs[:,M/2:]

train_outputs = outputs[:,:M/2]
test_outputs  = outputs[:,M/2:]

Ui,Si,Vi = np.linalg.svd(train_inputs)
en_f= 1 - np.cumsum(Si)/np.sum(Si)

acc = 0.99
r_f = np.argwhere(en_f<(1-acc))[0,0]

Uf = Ui[:,:r_f]
f_hat = np.matmul(Uf.T,train_inputs).T
fx = np.concatenate((np.repeat(f_hat,N,axis=0), np.tile(xgrid,M/2)[:,np.newaxis]),axis=1)
gx = np.reshape(train_outputs,((M/2)*N,),order='F')

Ndata = N*(M/2)

# load training indices
tr_i = np.load('tr_i.npy')

x_train = torch.from_numpy(fx[tr_i,:].astype(np.float32))
y_train = torch.from_numpy(gx[tr_i,np.newaxis].astype(np.float32))

N_neurons = 20

if N_neurons == 20:
    DirectNet = DirectNet_20
elif N_neurons == 50:
    DirectNet = DirectNet_50

model = DirectNet(r_f+1,1)
model = torch.load("fxNet_"+str(N_neurons)+".model")

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)

y_pred_train = model(x_train).detach().numpy().T
rel_err_nn_train = np.abs(y_pred_train-gx[tr_i])/np.abs(gx[tr_i])
mre_nn_train = np.mean(rel_err_nn_train)

mre_nn_tall = 0.
for i in range(N):
	y_pred_other = model(torch.from_numpy(fx[i*M/2:(i+1)*M/2,:].astype(np.float32))).detach().numpy()
	rel_err_nn_tall = np.abs(y_pred_other-gx)/np.abs(gx)
	mre_nn_tall += np.mean(rel_err_nn_tall)
	print mre_nn_tall

