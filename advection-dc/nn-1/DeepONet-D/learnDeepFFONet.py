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

compute_input_PCA = True


train_inputs = inputs[:, :M//2] 
x_train_part = train_inputs.T.astype(np.float32)

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


################################################################
# training and evaluation
################################################################

# batch_size = 1024



train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)


learning_rate = 0.001

epochs = 1000
step_size = 100
gamma = 0.5


model = DeepFFONet(r_f, 1, XY, layers,  layers, N_neurons) 
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

    torch.save(model, "DeepFFONetNet_" + str(N_neurons) + "_" + str(layers) + "Nd_" + str(ntrain) + ".model")
    scheduler.step()

    train_l2/= ntrain

    t2 = default_timer()
    print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)



print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)




