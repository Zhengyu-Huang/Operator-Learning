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
prefix = "../../../data/"  
a0 = np.load(prefix+"adv_a0.npy")
aT = np.load(prefix+"adv_aT.npy")

acc = 0.999

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]

inputs  = a0
outputs = aT

compute_input_PCA = True

if compute_input_PCA:
    train_inputs =  inputs[:, :M//2] 
    test_inputs  =  inputs[:, M//2:M] 
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]

    # r_f = min(r_f, 512)
    r_f = 200

    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)

    x_train_part = f_hat.T.astype(np.float32)
    x_test_part = f_hat_test.T.astype(np.float32)
else:
    
    print("must compute input PCA")

    

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



print("NN: ", N_neurons, "rel train error: ", mre_nn_train, "rel test error ", mre_nn_test)


# save smallest, medium, largest
test_input_save = np.zeros((N, 3))
test_output_save = np.zeros((N, 6))
for i, ind in enumerate([np.argmin(rel_err_nn_test), np.argsort(rel_err_nn_test)[len(rel_err_nn_test)//2], np.argmax(rel_err_nn_test)]):
    test_input_save[:, i] = inputs[:, M//2 + ind]
    # truth
    test_output_save[:, i] = outputs[:, M//2 + ind]
    # predict
    test_output_save[:, i + 3] = aT_test_pred[ind,:]

np.save(str(ntrain) + "_" + str(N_neurons) + "_test_input_save.npy", test_input_save)
np.save(str(ntrain) + "_" + str(N_neurons) + "_test_output_save.npy", test_output_save)