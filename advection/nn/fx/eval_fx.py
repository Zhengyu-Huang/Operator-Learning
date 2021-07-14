import sys
import numpy as np
sys.path.append('../../../nn')
from mynn import *
from datetime import datetime

import matplotlib as mpl 
from matplotlib.lines import Line2D 
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
# plt.rc("text", usetex=True)         # Crisp axis ticks
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


data    = np.load('../../data/data.npz')
inputs  = np.concatenate((data["theta"], data["u"][np.newaxis, :]), axis=0)
r_f, _ = inputs.shape
outputs = data["aT"]
a0 = data["a0"]
N, M = outputs.shape

xgrid = np.linspace(0,1,N+1)
xgrid = xgrid[:-1]
dx    = xgrid[1] - xgrid[0]



train_inputs = inputs[:,0::2]
test_inputs  = inputs[:,1::2]

train_outputs = outputs[:,0::2]
test_outputs  = outputs[:,1::2]

train_a0 = a0[:,0::2]
test_a0 = a0[:,1::2]

f_hat = train_inputs.T
fx = np.concatenate((np.repeat(f_hat,N,axis=0), np.tile(xgrid,M//2)[:,np.newaxis]),axis=1)
gx = np.reshape(train_outputs,((M//2)*N,),order='F')

Ndata = N*(M//2)

# load training indices
# tr_i = np.load('tr_i.npy')

x_train = torch.from_numpy(fx.astype(np.float32))
y_train = torch.from_numpy(gx[:,np.newaxis].astype(np.float32))

N_neurons = 50

if N_neurons == 20:
    DirectNet = DirectNet_20
elif N_neurons == 50:
    DirectNet = DirectNet_50

model = DirectNet(r_f+1,1)
model = torch.load("fxNet_"+str(N_neurons)+".model")

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)

y_pred_train = model(x_train).detach().numpy().flatten()

rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    rel_err_nn_train[i] = np.linalg.norm(y_pred_train[i*N:(i+1)*N] - gx[i*N:(i+1)*N])/np.linalg.norm(gx[i*N:(i+1)*N])
mre_nn_train = np.mean(rel_err_nn_train)

############# test

f_hat_test = test_inputs.T
fx_test = np.concatenate((np.repeat(f_hat_test,N,axis=0), np.tile(xgrid,M//2)[:,np.newaxis]),axis=1)
gx_test = np.reshape(test_outputs,((M//2)*N,),order='F')

# load training indices
# tr_i = np.load('tr_i.npy')

x_test = torch.from_numpy(fx_test.astype(np.float32))
y_pred_test = model(x_test).detach().numpy().flatten()

rel_err_nn_test = np.zeros(M//2)
for i in range(M//2):
    rel_err_nn_test[i] = np.linalg.norm(y_pred_test[i*N:(i+1)*N] - gx_test[i*N:(i+1)*N])/np.linalg.norm(gx_test[i*N:(i+1)*N])
mre_nn_test = np.mean(rel_err_nn_test)

print("NN: ", N_neurons, "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)

## Error plot
fig,ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(rel_err_nn_train,lw=0.5,color=color1,label='training')
ax.semilogy(rel_err_nn_test,lw=0.5,color=color2,label='test')
ax.legend()
plt.xlabel('data index')
plt.ylabel('Relative errors')
plt.savefig('NN%d_errors.png' %(N_neurons),pad_inches=3)
plt.close()

ind = np.argmax(rel_err_nn_test)

## worst case plot
fig,ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.plot(xgrid,test_a0[:,ind],'--',lw=0.5,color=color1,label='$u_0$')
ax.plot(xgrid,test_outputs[:,ind],lw=0.5,color=color2,label='$u(T)$')
ax.plot(xgrid,y_pred_test[ind*N:(ind+1)*N],lw=0.5,color=color3,label="NN u(T)")
ax.legend()
plt.xlabel('$x$')
plt.ylabel('u(x)')
plt.savefig('worst_case_test_NN%d.png' %(N_neurons),pad_inches=3)
plt.close()

# for ind in np.random.randint(0,1024,(5,)):
# 	fig,ax = plt.subplots(figsize=(3,3))
# 	fig.subplots_adjust(bottom=0.2,left = 0.15)
# 	ax.plot(xgrid,test_inputs[:,ind],'--',lw=0.5,color=color1,label='$u_0$')
# 	ax.plot(xgrid,test_outputs[:,ind],lw=0.5,color=color2,label='$u(T)$')
# 	ax.plot(xgrid,np.matmul(Ug,y_pred_test[:,ind]),lw=0.5,color=color3,label="NN u(T)")
# 	ax.legend()
# 	plt.xlabel('$x$')
# 	plt.ylabel('u(x)')
# 	plt.savefig('test'+str(ind)+'.png',pad_inches=3)
# 	plt.close()

ind = np.argmax(rel_err_nn_train)

fig,ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.plot(xgrid,train_a0[:,ind],'--',lw=0.5,color=color1,label='$u_0$')
ax.plot(xgrid,train_outputs[:,ind],lw=0.5,color=color2,label='$u(T)$')
ax.plot(xgrid,y_pred_train[ind*N:(ind+1)*N],lw=0.5,color=color3,label="NN u(T)")
ax.legend()
plt.xlabel('$x$')
plt.ylabel('u(x)')
plt.savefig('worst_case_train_NN%d.png' %(N_neurons),pad_inches=3)
plt.close()


# for ind in np.random.randint(0,1024,(9,)):
# 	fig,ax = plt.subplots(figsize=(3,3))
# 	fig.subplots_adjust(bottom=0.2,left = 0.15)
# 	ax.plot(xgrid,train_inputs[:,ind],'--',lw=0.5,color=color1,label='$u_0$')
# 	ax.plot(xgrid,train_outputs[:,ind],lw=0.5,color=color2,label='$u(T)$')
# 	ax.plot(xgrid,np.matmul(Ug,y_pred_train[:,ind]),lw=0.5,color=color3,label="NN u(T)")
# 	ax.legend()
# 	plt.xlabel('$x$')
# 	plt.ylabel('u(x)')
# 	plt.savefig('train'+str(ind)+'.png',pad_inches=3)
# 	plt.close()


# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.semilogy(en_f,lw=0.5,color=color1,label='input')
# ax.semilogy(en_g,lw=0.5,color=color2,label='output')
# ax.legend()
# plt.xlabel('index')
# plt.ylabel('energy lost')
# plt.savefig('training_PCA.png',pad_inches=3)
# plt.close()


