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

# N = 256
# K = 200
# M = 2048

# xgrid = np.linspace(0,1,N+1)
# xgrid = xgrid[:-1]
# dx    = xgrid[1] - xgrid[0]

# # burgers param and data
# nu      = 0.01
# data    = np.load('../../data/N'+str(N)+'_K'+str(K)+'_M'+str(M)+'.npz')
# inputs  = data["inputs"]
# outputs = data["outputs"]

T = 2
N = 128
K = 800
M = 2048
data    = np.load('../../data/T'+str(int(T))+'_N'+str(N)+'_K'+str(K)+'_M'+str(M)+'_traj2.npz')

traj = data['traj']
theta = data['data_theta']
inputs  = traj[:,0,:]
outputs = traj[:,-1,:]

xgrid = np.linspace(0,1,N+1)
xgrid = xgrid[:-1]
dx    = xgrid[1] - xgrid[0]

train_inputs = inputs[:,:M//2]
test_inputs  = inputs[:,M//2:]

train_outputs = outputs[:,:M//2]
test_outputs  = outputs[:,M//2:]

Ui,Si,Vi = np.linalg.svd(train_inputs)
en_f= 1 - np.cumsum(Si)/np.sum(Si)
Uo,So,Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So)/np.sum(So)

acc = 0.99
r_f = np.argwhere(en_f<(1-acc))[0,0]
r_g = np.argwhere(en_g<(1-acc))[0,0]

Uf = Ui[:,:r_f]
Ug = Uo[:,:r_g]

# best fit linear operator
f_hat = np.matmul(Uf.T,train_inputs)
g_hat = np.matmul(Ug.T,train_outputs)
temp, res, rnk,s = np.linalg.lstsq(f_hat.T,g_hat.T)
BFL = temp.T

err_train = np.matmul(BFL,f_hat)-g_hat
rel_err_train = np.sum(err_train**2,0)/np.sum(g_hat**2,0)
mean_rel_err_train = np.mean(rel_err_train)

g_hat_test = np.matmul(Ug.T,test_outputs)
err_test = np.matmul(BFL,np.matmul(Uf.T,test_inputs))-g_hat_test
rel_err_test = np.sum(err_test**2,0)/np.sum(g_hat_test**2,0)
mean_rel_err_test = np.mean(rel_err_test)


# neural network
x_train = torch.from_numpy(f_hat.T.astype(np.float32))
y_train = torch.from_numpy(g_hat.T.astype(np.float32))

N_neurons = 20

if N_neurons == 20:
    DirectNet = DirectNet_20
elif N_neurons == 50:
    DirectNet = DirectNet_50

model = torch.load("PCANet_"+str(N_neurons)+".model")

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)

y_pred_train = model(x_train).detach().numpy().T

rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    rel_err_nn_train[i] = np.linalg.norm(train_outputs[:, i]  - np.matmul(Ug, y_pred_train[:, i]))/np.linalg.norm(train_outputs[:, i])
mre_nn_train = np.mean(rel_err_nn_train)

# rel_err_nn_train = np.sum((y_pred_train-g_hat)**2,0)/np.sum(g_hat**2,0)
# mre_nn_train = np.mean(rel_err_nn_train)

f_hat_test = np.matmul(Uf.T,test_inputs)
y_pred_test  = model(torch.from_numpy(f_hat_test.T.astype(np.float32))).detach().numpy().T

rel_err_nn_test = np.zeros(M//2)
for i in range(M//2):
    rel_err_nn_test[i] = np.linalg.norm(test_outputs[:, i]  - np.matmul(Ug, y_pred_test[:, i]))/np.linalg.norm(test_outputs[:, i])
mre_nn_test = np.mean(rel_err_nn_test)

# rel_err_nn_test = np.sum((y_pred_test-g_hat_test)**2,0)/np.sum(g_hat_test**2,0)
# mre_nn_test = np.mean(rel_err_nn_test)
print("NN: ", N_neurons, "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)
# loss_scale = 1000
# n_epochs = 500000
# for epoch in range(n_epochs):
# 	y_pred = model(x_train)
# 	loss = loss_fn(y_pred,y_train)*loss_scale

# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()
# 	if epoch % 1000 == 0:
# 		print("[{}/{}], loss: {}, time {}".format(epoch, n_epochs, np.round(loss.item(), 3),datetime.now()))
# 		torch.save(model, "PCANet_"+str(N_neurons)+".model")

	
# # save the model
# torch.save(model, "PCANet_"+str(N_neurons)+".model")

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


fig,ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.plot(xgrid,test_inputs[:,ind],'--',lw=0.5,color=color1,label='$u_0$')
ax.plot(xgrid,test_outputs[:,ind],lw=0.5,color=color2,label='$u(T)$')
ax.plot(xgrid,np.matmul(Ug,y_pred_test[:,ind]),lw=0.5,color=color3,label="NN u(T)")
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
ax.plot(xgrid,train_inputs[:,ind],'--',lw=0.5,color=color1,label='$u_0$')
ax.plot(xgrid,train_outputs[:,ind],lw=0.5,color=color2,label='$u(T)$')
ax.plot(xgrid,np.matmul(Ug,y_pred_train[:,ind]),lw=0.5,color=color3,label="NN u(T)")
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
