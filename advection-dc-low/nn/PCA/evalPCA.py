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


M = int(sys.argv[1])
N_neurons = int(sys.argv[2])
layers    = int(sys.argv[3])
batch_size = int(sys.argv[4])


N = 200
ntrain = M//2
N_theta = 100
prefix = "../../../data/"  
a0 = np.load(prefix+"adv_a0.npy")
aT = np.load(prefix+"adv_aT.npy")

acc=0.999

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]

inputs  = a0
outputs = aT


compute_input_PCA = True

if compute_input_PCA:
    train_inputs = inputs[:,:M//2] 
    test_inputs  = inputs[:, M//2:M]
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]

    r_f = 200
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))
else:
    
    print("must compute input PCA")
    


train_outputs = outputs[:,:M//2] 
test_outputs  = outputs[:,M//2:M]
Uo,So,Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So)/np.sum(So)
r_g = np.argwhere(en_g<(1-acc))[0,0]
Ug = Uo[:,:r_g]
g_hat = np.matmul(Ug.T,train_outputs) 
y_train = torch.from_numpy(g_hat.T.astype(np.float32))



model = torch.load("PCANet_" + str(N_neurons) + "_" + str(layers) + "Nd_" + str(ntrain) + ".model")
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


x_test = torch.from_numpy(f_hat_test.T.astype(np.float32))
x_test = x_normalizer.encode(x_test.to(device))
y_pred_test  = y_normalizer.decode(model(x_test).detach()).cpu().numpy().T

rel_err_nn_test = np.zeros(M//2)
for i in range(M//2):
    rel_err_nn_test[i] = np.linalg.norm(test_outputs[:, i]  - np.matmul(Ug, y_pred_test[:, i]))/np.linalg.norm(test_outputs[:, i])
mre_nn_test = np.mean(rel_err_nn_test)

print("NN: ", N_neurons, "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)


#########################################
# save smallest, medium, largest
test_input_save = np.zeros((N, 3))
test_output_save = np.zeros((N, 6))
for i, ind in enumerate([np.argmin(rel_err_nn_test), np.argsort(rel_err_nn_test)[len(rel_err_nn_test)//2], np.argmax(rel_err_nn_test)]):
    test_input_save[:, i] = inputs[:, M//2 + ind]
    # truth
    test_output_save[:, i] = outputs[:, M//2 + ind]
    # predict
    test_output_save[:, i + 3] =  np.matmul(Ug, y_pred_test[:, ind])

np.save(str(ntrain) + "_" + str(N_neurons) + "_test_input_save.npy", test_input_save)
np.save(str(ntrain) + "_" + str(N_neurons) + "_test_output_save.npy", test_output_save)
