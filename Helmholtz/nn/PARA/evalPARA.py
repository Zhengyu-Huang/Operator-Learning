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


def colnorm(u):
	return np.sqrt(np.sum(u**2,0))



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

N = 100
M = 100
N_theta = 100
prefix = "../"
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

    r_f = min(r_f, 500)
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
x_train = np.zeros((M//2 * N_upper, r_f + 2))
y_train = np.zeros(M//2 * N_upper)

for i in range(M//2):
    d_range = range(i*N_upper, (i + 1)*N_upper)
    x_train[d_range , 0:r_f]   = x_train_part[i, :]
    x_train[d_range , r_f]     = X_upper
    x_train[d_range , r_f + 1] = Y_upper 
    y_train[d_range] = full2upper(K[:, :, i])

x_test = np.zeros(((M-M//2) * N_upper, r_f + 2))

for i in range(M-M//2):
    d_range = range(i*N_upper, (i + 1)*N_upper)
    x_test[d_range , 0:r_f]   = x_test_part[i, :]
    x_test[d_range , r_f]     = X_upper
    x_test[d_range , r_f + 1] = Y_upper 
    
      


print("Input dim : ", r_f+2, " output dim : ", 1)
 

N_neurons = 100
model = torch.load("PARANet_"+str(N_neurons)+".model", map_location=device)
model.to(device)

# model = torch.load("PCANet_"+str(N_neurons)+".model")

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)

loss_scale = 1000
n_epochs = 500

x_train = torch.from_numpy(x_train.astype(np.float32)).to(device)
x_test = torch.from_numpy(x_test.astype(np.float32)).to(device)
# y_pred = y_pred.to(device)

# Training error
K_train_pred_upper = model(x_train).detach().cpu().numpy()
K_train_pred = upper2full(K_train_pred_upper, M//2)


rel_err_nn_train = np.zeros(M//2)
for i in np.arange(0, M//2):
    rel_err_nn_train[i] =  np.linalg.norm(K_train_pred[:, :, i] - K_train[:, :, i])/np.linalg.norm(K_train[:, :, i])
mre_nn_train = np.mean(rel_err_nn_train)




# Test error
K_test_pred_upper = model(x_test).detach().cpu().numpy()
K_test_pred = upper2full(K_test_pred_upper, M//2)
rel_err_nn_test = np.zeros(M//2)
for i in np.arange(0, M//2):
    rel_err_nn_test[i] =  np.linalg.norm(K_test_pred[:, :, i] - K_test[:, :, i])/np.linalg.norm(K_test[:, :, i])
mre_nn_test = np.mean(rel_err_nn_test)

print("NN: ", N_neurons, "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)



fig,ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(rel_err_nn_train,lw=0.5,color=color1,label='training')
ax.semilogy(rel_err_nn_test,lw=0.5,color=color2,label='test')
ax.legend()
plt.xlabel('data index')
plt.ylabel('Relative errors')
plt.savefig('NN%d_errors.png' %(N_neurons),pad_inches=3)
plt.close()


####### worst error plot
ind = np.argmax(rel_err_nn_test)

fig,ax = plt.subplots(ncols=3, figsize=(9,3))
vmin, vmax = K_train[:,:,ind].min(), K_train[:,:,ind].max()
ax[0].pcolormesh(X, Y, np.reshape(test_inputs[:, ind], (N+1,N+1)),  shading='gouraud')
ax[1].pcolormesh(X, Y, K_train_pred[:,:,ind], shading='gouraud', vmin=vmin, vmax =vmax)
ax[2].pcolormesh(X, Y, K_train[:,:,ind], shading='gouraud', vmin=vmin, vmax =vmax)
plt.xlabel('$x$')
plt.ylabel('u(x)')
plt.savefig('worst_case_train_NN%d.png' %(N_neurons),pad_inches=3)
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

fig,ax = plt.subplots(ncols=3, figsize=(9,3))
vmin, vmax = K_train[:,:,ind].min(), K_train[:,:,ind].max()
ax[0].pcolormesh(X, Y, np.reshape(test_inputs[:, ind], (N+1,N+1)),  shading='gouraud')
ax[1].pcolormesh(X, Y, K_test_pred[:,:,ind], shading='gouraud', vmin=vmin, vmax =vmax)
ax[2].pcolormesh(X, Y, K_test[:,:,ind], shading='gouraud', vmin=vmin, vmax =vmax)
plt.xlabel('$x$')
plt.ylabel('u(x)')
plt.savefig('worst_case_test_NN%d.png' %(N_neurons),pad_inches=3)
plt.close()
