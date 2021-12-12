import sys
import numpy as np
sys.path.append('../../../nn')
from mynn import *
from mydata import *
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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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


M = int(sys.argv[1]) #5000
N_neurons = int(sys.argv[2])


N = 21
ntrain = M//2
N_theta = 100
prefix = "/central/scratch/dzhuang/Helmholtz_data/"
theta = np.load(prefix+"Random_UnitCell_theta_" + str(N_theta) + ".npy")   
K = np.load(prefix+"Random_UnitCell_sigma_" + str(N_theta) + ".npy")
cs = np.load(prefix+"Random_UnitCell_Fn_" + str(N_theta) + ".npy")

acc=0.999

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]

inputs  = cs
outputs = K


compute_input_PCA = True

if compute_input_PCA:
    train_inputs = inputs[:,:M//2]
    test_inputs  = inputs[:,M//2:M]
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    # r_f = min(r_f, 512)

    r_f = 21
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
else:
    
    train_inputs =  theta[:M//2, :]
    test_inputs  = theta[M//2:M, :]
    r_f = N_theta
    x_train = torch.from_numpy(train_inputs.astype(np.float32))
    


train_outputs = outputs[:,:M//2] 
test_outputs  = outputs[:,M//2:M] 
Uo,So,Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So)/np.sum(So)
r_g = np.argwhere(en_g<(1-acc))[0,0]
Ug = Uo[:,:r_g]
g_hat = np.matmul(Ug.T,train_outputs) 
y_train = torch.from_numpy(g_hat.T.astype(np.float32))



model = torch.load("PCANet_"+str(N_neurons)+"Nd_"+str(ntrain)+".model")
model.to(device)


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

if torch.cuda.is_available():
    x_normalizer.cuda()
    y_normalizer.cuda()

x_train = x_train.to(device)
y_pred_train = y_normalizer.decode(model(x_train).detach()).cpu().numpy().T

rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    rel_err_nn_train[i] = np.linalg.norm(train_outputs[:, i]  - np.matmul(Ug, y_pred_train[:, i]))/np.linalg.norm(train_outputs[:, i])
mre_nn_train = np.mean(rel_err_nn_train)

# rel_err_nn_train = np.sum((y_pred_train-g_hat)**2,0)/np.sum(g_hat**2,0)
# mre_nn_train = np.mean(rel_err_nn_train)

# print(f_hat_test.shape)
# f_hat_test = np.matmul(Uf.T,test_inputs)
x_test = torch.from_numpy(f_hat_test.T.astype(np.float32))
x_test = x_normalizer.encode(x_test.to(device))
y_pred_test  = y_normalizer.decode(model(x_test).detach()).cpu().numpy().T

rel_err_nn_test = np.zeros(M//2)
for i in range(M//2):
    rel_err_nn_test[i] = np.linalg.norm(test_outputs[:, i]  - np.matmul(Ug, y_pred_test[:, i]))/np.linalg.norm(test_outputs[:, i])
mre_nn_test = np.mean(rel_err_nn_test)

# rel_err_nn_test = np.sum((y_pred_test-g_hat_test)**2,0)/np.sum(g_hat_test**2,0)
# mre_nn_test = np.mean(rel_err_nn_test)
print("NN: ", N_neurons, "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)


fig,ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(rel_err_nn_train,lw=0.5,color=color1,label='training')
ax.semilogy(rel_err_nn_test,lw=0.5,color=color2,label='test')
ax.legend()
plt.xlabel('data index')
plt.ylabel('Relative errors')
plt.tight_layout()
plt.savefig('NN%d_errors.png' %(N_neurons),pad_inches=3)
plt.close()

'''
ind = np.argmax(rel_err_nn_test)
Y, X = np.meshgrid(xgrid, xgrid)

fig,ax = plt.subplots(ncols=3, figsize=(9,3))
vmin, vmax = min(test_outputs[:,ind]), max(test_outputs[:,ind])
ax[0].pcolormesh(X, Y, np.reshape(test_inputs[:,ind],(N+1,N+1)),               shading='gouraud')
ax[1].pcolormesh(X, Y, np.reshape(test_outputs[:,ind],(N+1,N+1)),              shading='gouraud', vmin=vmin, vmax =vmax)
ax[2].pcolormesh(X, Y, np.reshape(np.matmul(Ug,y_pred_test[:,ind]),(N+1,N+1)), shading='gouraud', vmin=vmin, vmax =vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('worst_case_test_NN%d.png' %(N_neurons),pad_inches=3)
plt.close()


ind = np.argmax(rel_err_nn_train)

fig,ax = plt.subplots(ncols=3, figsize=(9,3))
vmin, vmax = min(test_outputs[:,ind]), max(test_outputs[:,ind])
ax[0].pcolormesh(X, Y, np.reshape(train_inputs[:,ind],(N+1,N+1)),               shading="gouraud")
ax[1].pcolormesh(X, Y, np.reshape(train_outputs[:,ind],(N+1,N+1)),              shading="gouraud", vmin=vmin, vmax =vmax)
ax[2].pcolormesh(X, Y, np.reshape(np.matmul(Ug,y_pred_train[:,ind]),(N+1,N+1)), shading="gouraud", vmin=vmin, vmax =vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('worst_case_train_NN%d.png' %(N_neurons),pad_inches=3)
plt.close()

'''
