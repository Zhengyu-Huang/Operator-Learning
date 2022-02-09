import sys
import numpy as np
sys.path.append('../../../nn')
from mynn import *
from mydata import *
from Adam import Adam


import operator
from functools import reduce
from functools import partial

from timeit import default_timer


torch.manual_seed(0)
np.random.seed(0)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

M = int(sys.argv[1]) #5000
N_neurons = int(sys.argv[2])

N = 21

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

compute_input_PCA = True

if compute_input_PCA:
    train_inputs = inputs[:,:M//2]
    # test_inputs  = np.reshape(inputs[:,:,M//2:M], (-1, M-M//2))
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    
    # r_f = min(r_f, 512)
    r_f = 21
    # r_f = 512
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    x_train_part = f_hat.T.astype(np.float32)
else:
    
    print("must use compute_input_PCA")


del train_inputs
del inputs
del Ui, Vi, Uf, f_hat


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
x_train  = torch.from_numpy(x_train)
y_train  = torch.from_numpy(y_train)

x_normalizer = UnitGaussianNormalizer(x_train)
x_train      = x_normalizer.encode(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
y_train      = y_normalizer.encode(y_train)


################################################################
# training and evaluation
################################################################

batch_size = 16

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)


learning_rate = 0.001

epochs = 1000
step_size = 100
gamma = 0.5




layers = 4
model = DeepFFONet(r_f, 2, XY_upper, layers,  layers, N_neurons) 
print(count_params(model))
model.to(device)


optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = torch.nn.MSELoss(reduction='sum')
y_normalizer.cuda()
t0 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        batch_size_ = x.shape[0]
        optimizer.zero_grad()
        out = model(x)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out , y)
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    torch.save(model, "DeepFFONetNet_"+str(N_neurons)+"Nd_"+str(ntrain)+".model")
    scheduler.step()

    train_l2/= ntrain

    t2 = default_timer()
    print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)



print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)




