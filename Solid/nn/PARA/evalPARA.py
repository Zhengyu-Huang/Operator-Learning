import sys
import numpy as np
sys.path.append('../../../nn')
from mynn import *
from mydata import *
from Adam import Adam
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


def colnorm(u):
	return np.sqrt(np.sum(u**2,0))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

M = int(sys.argv[1]) #5000
N_neurons = int(sys.argv[2])
batch_size = 1024 #int(sys.argv[3])

N = 21

ntrain = M//2
N_theta = 100
prefix = "/central/scratch/dzhuang/Helmholtz_data/"

theta = np.load(prefix+"Random_UnitCell_theta_" + str(N_theta) + ".npy")   
K = np.load(prefix+"Random_UnitCell_sigma_" + str(N_theta) + ".npy")
cs = np.load(prefix+"Random_UnitCell_Fn_" + str(N_theta) + ".npy")
XY = np.load(prefix+"Random_UnitCell_XY_" + str(N_theta) + ".npy")

K_train = K[:,  :M//2]
K_test = K[:, M//2:M]
acc = 0.999

inputs  = cs
outputs = K

compute_input_PCA = True

if compute_input_PCA:
    train_inputs =  inputs[:,:M//2] 
    test_inputs  =  inputs[:,M//2:M] 
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]

    # r_f = min(r_f, 512)
    r_f = 21
    # r_f = 512
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)

    x_train_part = f_hat.T.astype(np.float32)
    x_test_part = f_hat_test.T.astype(np.float32)
else:
    
    train_inputs =  theta[:M//2, :]
    test_inputs  = theta[M//2:M, :]
    r_f = N_theta
    x_train_part = train_inputs.astype(np.float32)
    x_test_part = test_inputs.astype(np.float32)

    
del inputs
del Ui, Vi, Uf, f_hat


X_upper = XY[:,0]
Y_upper = XY[:,1]
N_upper = XY.shape[0]
x_train = np.zeros((M//2 * N_upper, r_f + 2), dtype = np.float32)
y_train = np.zeros(M//2 * N_upper, dtype = np.float32)

for i in range(M//2):
    d_range = range(i*N_upper, (i + 1)*N_upper)
    x_train[d_range , 0:r_f]   = x_train_part[i, :]
    x_train[d_range , r_f]     = X_upper
    x_train[d_range , r_f + 1] = Y_upper 
    y_train[d_range] = K[:, i]

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).unsqueeze(-1)

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)


      
print("Input dim : ", r_f+2, " output dim : ", 1)
 
model = torch.load("PARANet_"+str(N_neurons)+"Nd_"+str(ntrain)+ "Nb_"+str(batch_size)+".model", map_location=device)
model.to(device)

if torch.cuda.is_available():
    y_normalizer.cuda()

# Training error
rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    print("i / N = ", i, " / ", M//2)
    K_train_pred = y_normalizer.decode(model( x_train[i*N_upper:(i+1)*N_upper, :].to(device) )).detach().cpu().numpy()
    
    rel_err_nn_train[i] =  np.linalg.norm(K_train_pred.squeeze() - K_train[:, i])/np.linalg.norm(K_train[:, i])
mre_nn_train = np.mean(rel_err_nn_train)

# ####### worst error plot
# i = np.argmax(rel_err_nn_train)
# K_train_pred_upper = y_normalizer.decode(model(x_train[i*N_upper:(i+1)*N_upper, :].to(device) )).detach().cpu().numpy()
# K_train_pred = upper2full_1(K_train_pred_upper)
# fig,ax = plt.subplots(ncols=3, figsize=(9,3))
# vmin, vmax = K_train[:,:,i].min(), K_train[:,:,i].max()
# ax[0].pcolormesh(X, Y, np.reshape(test_inputs[:, i], (N+1,N+1)),  shading='gouraud')
# ax[1].pcolormesh(X, Y, K_train_pred, shading='gouraud', vmin=vmin, vmax =vmax)
# ax[2].pcolormesh(X, Y, K_train[:,:,i], shading='gouraud', vmin=vmin, vmax =vmax)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.tight_layout()
# plt.savefig('worst_case_train_NN%d.png' %(N_neurons),pad_inches=3)
# plt.close()


del x_train,  K_train
########### Test
x_test = np.zeros(((M-M//2) * N_upper, r_f + 2), dtype = np.float32)
for i in range(M-M//2):
    d_range = range(i*N_upper, (i + 1)*N_upper)
    x_test[d_range , 0:r_f]   = x_test_part[i, :]
    x_test[d_range , r_f]     = X_upper
    x_test[d_range , r_f + 1] = Y_upper 

x_test = torch.from_numpy(x_test)
x_test = x_normalizer.encode(x_test)

# Test error
rel_err_nn_test = np.zeros(M//2)
for i in range(M-M//2):
    print("i / N = ", i, " / ", M-M//2)
    K_test_pred = y_normalizer.decode(model(x_test[i*N_upper:(i+1)*N_upper, :].to(device) )).detach().cpu().numpy()
    rel_err_nn_test[i] =  np.linalg.norm(K_test_pred.squeeze() - K_test[:, i])/np.linalg.norm(K_test[:, i])
mre_nn_test = np.mean(rel_err_nn_test)

# ####### worst error plot
# i = np.argmax(rel_err_nn_test)
# K_test_pred_upper = y_normalizer.decode(model(x_test[i*N_upper:(i+1)*N_upper, :].to(device) )).detach().cpu().numpy()
# K_test_pred = upper2full_1(K_test_pred_upper)
# fig,ax = plt.subplots(ncols=3, figsize=(9,3))
# vmin, vmax = K_test[:,:,i].min(), K_test[:,:,i].max()
# ax[0].pcolormesh(X, Y, np.reshape(test_inputs[:, i], (N+1,N+1)),  shading='gouraud')
# ax[1].pcolormesh(X, Y, K_test_pred, shading='gouraud', vmin=vmin, vmax =vmax)
# ax[2].pcolormesh(X, Y, K_test[:,:,i], shading='gouraud', vmin=vmin, vmax =vmax)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.tight_layout()
# plt.savefig('worst_case_test_NN%d.png' %(N_neurons),pad_inches=3)
# plt.close()


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

print("NN: ", N_neurons, "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)
