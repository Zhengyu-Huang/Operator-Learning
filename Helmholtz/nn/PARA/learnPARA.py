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
    x_train_part = f_hat.T.astype(np.float32)
else:
    
    train_inputs =  theta[:M//2, :]
    test_inputs  = theta[M//2:M, :]
    r_f = N_theta
    x_train_part = train_inputs.astype(np.float32)
    

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
   


print("Input dim : ", r_f+2, " output dim : ", 1)
 

N_neurons = 100
layers = 4
model = FNN(r_f + 2, 1, layers, N_neurons) 
model.to(device)

# model = torch.load("PCANet_"+str(N_neurons)+".model")

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)

loss_scale = 1000
n_epochs = 500

x_train = torch.from_numpy(x_train.astype(np.float32)).to(device)
y_train = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(-1).to(device)
# y_pred = y_pred.to(device)


ds = DirectData(X=x_train, y=y_train)
ds = DataLoader(ds, batch_size=512, shuffle=True)


for epoch in range(n_epochs):

    for ix, (_x, _y) in enumerate(ds):
        # _x, _y = _x.to(device), _y.to(device)
        y_pred = model(_x)
        loss = loss_fn(y_pred,_y)*loss_scale
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("[{}/{}], loss: {}, time {}".format(epoch, n_epochs, np.round(loss.item(), 3),datetime.now()))
        torch.save(model, "PARANet_"+str(N_neurons)+".model")

	
# save the model
torch.save(model, "PARANet_"+str(N_neurons)+".model")

# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.plot(rel_err_train,lw=0.5,color=color1,label='training')
# ax.plot(rel_err_test,lw=0.5,color=color2,label='test')
# ax.legend()
# plt.xlabel('data index')
# plt.ylabel('Relative errors')
# plt.savefig('LLS_errors.png',pad_inches=3)
# plt.close()

# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.semilogy(en_f,lw=0.5,color=color1,label='input')
# ax.semilogy(en_g,lw=0.5,color=color2,label='output')
# ax.legend()
# plt.xlabel('index')
# plt.ylabel('energy lost')
# plt.savefig('training_PCA.png',pad_inches=3)
# plt.close()
