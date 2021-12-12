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

M         = int(sys.argv[1]) 
N_neurons = int(sys.argv[2])
layers    = int(sys.argv[3])
batch_size = int(sys.argv[4])

N = 200
ntrain = M//2
N_theta = 100
prefix = "/home/dzhuang/Helmholtz-Data/advection-dc/src/"  
a0 = np.load(prefix+"adv_a0.npy")
aT = np.load(prefix+"adv_aT.npy")

acc = 0.999

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]

inputs  = a0
outputs = aT

train_inputs = inputs[:, :M//2] 
x_train_part = train_inputs.T.astype(np.float32)


test_inputs = inputs[:, M//2:M] 
x_test_part = test_inputs.T.astype(np.float32)


r_f = N

x_train = np.zeros((M//2, r_f), dtype = np.float32)
y_train = np.zeros((M//2, N), dtype = np.float32)

for i in range(M//2):
    y_train[i,:] = aT[:, i]
  

x_train = x_train_part
print("Input dim : ", r_f, " output dim : ", N)
 


XY = torch.from_numpy(np.reshape(xgrid, (N, 1)).astype(np.float32)).to(device)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

x_normalizer = UnitGaussianNormalizer(x_train)
x_normalizer.encode_(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
y_normalizer.encode_(y_train)


if torch.cuda.is_available():
    y_normalizer.cuda()
      
print("Input dim : ", r_f, " output dim : ", N)
 

model = torch.load("DeepFFONetNet_" + str(N_neurons) + "_" + str(layers) + "Nd_" + str(ntrain) + ".model")
model.to(device)


aT_train_pred = y_normalizer.decode(model(x_train.to(device) ).detach()).cpu().numpy()
# Training error
rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    print("i / N = ", i, " / ", M//2)
    rel_err_nn_train[i] =  np.linalg.norm(aT_train_pred[i,:] - aT[:, i])/np.linalg.norm(aT[:, i])
mre_nn_train = np.mean(rel_err_nn_train)

####### worst error plot
# i = np.argmax(rel_err_nn_train)
# K_train_pred = upper2full_1(K_train_pred_upper[i,:])
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

########### Test
x_test = x_test_part
# x_normalizer.cpu()
x_test = torch.from_numpy(x_test)
x_normalizer.encode_(x_test)


aT_test_pred = y_normalizer.decode(model(x_test.to(device)).detach()).cpu().numpy()
# Test error
rel_err_nn_test = np.zeros(M//2)
for i in range(M-M//2):
    print("i / N = ", i, " / ", M-M//2)
    rel_err_nn_test[i] =  np.linalg.norm(aT_test_pred[i,:] - aT[:, i+M//2])/np.linalg.norm(aT[:, i])
mre_nn_test = np.mean(rel_err_nn_test)

####### worst error plot
# i = np.argmax(rel_err_nn_test)
# K_test_pred = upper2full_1(K_test_pred_upper[i,:])
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


# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.semilogy(rel_err_nn_train,lw=0.5,color=color1,label='training')
# ax.semilogy(rel_err_nn_test,lw=0.5,color=color2,label='test')
# ax.legend()
# plt.xlabel('data index')
# plt.ylabel('Relative errors')
# plt.tight_layout()
# plt.savefig('NN%d_errors.png' %(N_neurons),pad_inches=3)
# plt.close()

print("NN: ", N_neurons, "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)
