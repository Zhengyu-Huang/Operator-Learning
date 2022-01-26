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

M = int(sys.argv[1])
N_neurons = int(sys.argv[2])
layers    = int(sys.argv[3])
batch_size = int(sys.argv[4])


N = 100
ntrain = M//2
N_theta = 100
prefix = "/central/scratch/dzhuang/Helmholtz_data/"
theta = np.load(prefix+"Random_Helmholtz_theta_" + str(N_theta) + ".npy")   
K = np.load(prefix+"Random_Helmholtz_K_" + str(N_theta) + ".npy")
cs = np.load(prefix+"Random_Helmholtz_cs_" + str(N_theta) + ".npy")



K_train = K[:, :, :M//2]
K_test = K[:, :, M//2:M]
acc = 0.99

xgrid = np.linspace(0,1,N+1)
dx    = xgrid[1] - xgrid[0]

inputs  = cs
outputs = K

compute_input_PCA = True

if compute_input_PCA:
    train_inputs = np.reshape(inputs[:,:,:M//2], (-1, M//2))
    test_inputs  = np.reshape(inputs[:,:,M//2:M], (-1, M-M//2))
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]

    # r_f = min(r_f, 512)
    r_f = 512

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



Y, X = np.meshgrid(xgrid, xgrid)
# test
i = 20
j = 40
assert(X[i, j] == i*dx and Y[i, j] == j*dx)

X_upper = full2upper(X)
Y_upper = full2upper(Y)
N_upper = len(X_upper)
x_train = np.zeros((M//2, r_f), dtype = np.float32)
y_train = np.zeros((M//2, N_upper), dtype = np.float32)

for i in range(M//2):
    y_train[i] = full2upper(K[:, :, i])
  
x_train = x_train_part
XY_upper = np.vstack((X_upper, Y_upper)).T

print("Input dim : ", r_f, " output dim : ", N_upper)

XY_upper = torch.from_numpy(XY_upper.astype(np.float32)).to(device)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


x_normalizer = UnitGaussianNormalizer(x_train)
x_normalizer.encode_(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)


if torch.cuda.is_available():
    y_normalizer.cuda()
      
print("Input dim : ", r_f+2, " output dim : ", N_upper)
 

model = torch.load("DeepFFONetNet_" + str(N_neurons) + "_" + str(layers) + "Nd_" + str(ntrain) + ".model")
model.to(device)


K_train_pred_upper = y_normalizer.decode(model(x_train.to(device) ).detach()).cpu().numpy()
# Training error
rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    print("i / N = ", i, " / ", M//2)
    K_train_pred = upper2full_1(K_train_pred_upper[i, :])
    rel_err_nn_train[i] =  np.linalg.norm(K_train_pred - K_train[:, :, i])/np.linalg.norm(K_train[:, :, i])
mre_nn_train = np.mean(rel_err_nn_train)

####### worst error plot
i = np.argmax(rel_err_nn_train)
K_train_pred = upper2full_1(K_train_pred_upper[i,:])
fig,ax = plt.subplots(ncols=3, figsize=(9,3))
vmin, vmax = K_train[:,:,i].min(), K_train[:,:,i].max()
ax[0].pcolormesh(X, Y, np.reshape(test_inputs[:, i], (N+1,N+1)),  shading='gouraud')
ax[1].pcolormesh(X, Y, K_train_pred, shading='gouraud', vmin=vmin, vmax =vmax)
ax[2].pcolormesh(X, Y, K_train[:,:,i], shading='gouraud', vmin=vmin, vmax =vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('worst_case_train_NN%d.png' %(N_neurons),pad_inches=3)
plt.close()


del x_train,  K_train
########### Test
x_test = x_test_part
# x_normalizer.cpu()
x_test = torch.from_numpy(x_test)
x_normalizer.encode_(x_test)


K_test_pred_upper = y_normalizer.decode(model(x_test.to(device)).detach()).cpu().numpy()
# Test error
rel_err_nn_test = np.zeros(M//2)
for i in range(M-M//2):
    print("i / N = ", i, " / ", M-M//2)
    K_test_pred = upper2full_1(K_test_pred_upper[i,:])
    rel_err_nn_test[i] =  np.linalg.norm(K_test_pred - K_test[:, :, i])/np.linalg.norm(K_test[:, :, i])
mre_nn_test = np.mean(rel_err_nn_test)

####### worst error plot
i = np.argmax(rel_err_nn_test)
K_test_pred = upper2full_1(K_test_pred_upper[i,:])
fig,ax = plt.subplots(ncols=3, figsize=(9,3))
vmin, vmax = K_test[:,:,i].min(), K_test[:,:,i].max()
ax[0].pcolormesh(X, Y, np.reshape(test_inputs[:, i], (N+1,N+1)),  shading='gouraud')
ax[1].pcolormesh(X, Y, K_test_pred, shading='gouraud', vmin=vmin, vmax =vmax)
ax[2].pcolormesh(X, Y, K_test[:,:,i], shading='gouraud', vmin=vmin, vmax =vmax)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig('worst_case_test_NN%d.png' %(N_neurons),pad_inches=3)
plt.close()


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



#########################################
# save smallest, medium, largest
test_input_save  = np.zeros((N+1,  N+1, 3))
test_output_save = np.zeros((N+1,  N+1, 6))
for i, ind in enumerate([np.argmin(rel_err_nn_test), np.argsort(rel_err_nn_test)[len(rel_err_nn_test)//2], np.argmax(rel_err_nn_test)]):
    test_input_save[:, :, i] = inputs[:, :, M//2 + ind]
    # truth
    test_output_save[:, :, i] = outputs[:, :, M//2 + ind]
    # predict
    K_test_pred = upper2full_1(K_test_pred_upper[ind,:])
    test_output_save[:, :, i + 3] =  K_test_pred

np.save(str(ntrain) + "_" + str(N_neurons) + "_test_input_save.npy", test_input_save)
np.save(str(ntrain) + "_" + str(N_neurons) + "_test_output_save.npy", test_output_save)


