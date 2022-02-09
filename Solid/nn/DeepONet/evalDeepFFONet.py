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

M = int(sys.argv[1]) #5000
N_neurons = int(sys.argv[2])

N = 21  # element

ntrain = M//2
N_theta = 100
prefix = "../../../data/"
theta = np.load(prefix+"Random_UnitCell_theta_" + str(N_theta) + ".npy")   
K = np.load(prefix+"Random_UnitCell_sigma_" + str(N_theta) + ".npy")
cs = np.load(prefix+"Random_UnitCell_Fn_" + str(N_theta) + ".npy")
XY = np.load(prefix+"Random_UnitCell_XY_" + str(N_theta) + ".npy")


acc = 0.999

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]

inputs  = cs
outputs = K

K_train = K[:, 0 : M//2]
K_test = K[:,  M//2 : M]

compute_input_PCA = True

if compute_input_PCA:
    train_inputs = inputs[:,:M//2]
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
    
    print("must use compute_input_PCA")
    

# Y, X = np.meshgrid(xgrid, xgrid)
# # test
# i = 20
# j = 40
# assert(X[i, j] == i*dx and Y[i, j] == j*dx)

X_upper = XY[:,0]
Y_upper = XY[:,1]
N_upper = XY.shape[0]
x_train = x_train_part
y_train = np.zeros((M//2 , N_upper), dtype = np.float32)

for i in range(M//2):
    y_train[i, :] = K[:, i]
    
XY_upper = np.vstack((X_upper, Y_upper)).T    

print("Input dim : ", r_f, " output dim : ", N_upper)



XY_upper = torch.from_numpy(XY_upper.astype(np.float32)).to(device)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

if torch.cuda.is_available():
    y_normalizer.cuda()
      
 

model = torch.load("DeepFFONetNet_"+str(N_neurons)+"Nd_"+str(ntrain)+".model", map_location=device)
model.to(device)

# Training error
K_train_pred = y_normalizer.decode(model( x_train.to(device) ).detach()).cpu().numpy()
rel_err_nn_train = np.zeros(M//2)
for i in range(M//2):
    rel_err_nn_train[i] =  np.linalg.norm(K_train_pred[i, :] - K_train[:, i])/np.linalg.norm(K_train[:, i])
mre_nn_train = np.mean(rel_err_nn_train)




# del x_train,  K_train
########### Test
x_test = x_test_part
# x_normalizer.cpu()
x_test = x_normalizer.encode(torch.from_numpy(x_test)) 
# Test error
rel_err_nn_test = np.zeros(M//2)
K_test_pred = y_normalizer.decode(model(x_test.to(device)).detach()).cpu().numpy()
for i in range(M-M//2):
    rel_err_nn_test[i] =  np.linalg.norm(K_test_pred[i, :] - K_test[:, i])/np.linalg.norm(K_test[:, i])
mre_nn_test = np.mean(rel_err_nn_test)


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
test_input_save  = np.zeros((inputs.shape[0],  3))
test_output_save = np.zeros((outputs.shape[0],  6))
for i, ind in enumerate([np.argmin(rel_err_nn_test), np.argsort(rel_err_nn_test)[len(rel_err_nn_test)//2], np.argmax(rel_err_nn_test)]):
    test_input_save[:, i]  = inputs[:, M//2 + ind]
    # truth
    test_output_save[:, i] = outputs[:, M//2 + ind]
    # predict
    test_output_save[:, i + 3] =  K_test_pred[ind, :]

np.save(str(ntrain) + "_" + str(N_neurons) + "_test_input_save.npy",  test_input_save)
np.save(str(ntrain) + "_" + str(N_neurons) + "_test_output_save.npy", test_output_save)
