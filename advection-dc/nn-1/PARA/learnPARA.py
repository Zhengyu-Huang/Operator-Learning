import sys
import numpy as np
sys.path.append('../../../nn')
from mynn import *
from mydata import *
from Adam import Adam
from timeit import default_timer


torch.manual_seed(0)
np.random.seed(0)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

M = int(sys.argv[1]) #5000
N_neurons = int(sys.argv[2])

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

if compute_input_PCA:
    train_inputs =  inputs[:,:M//2]
    # test_inputs  = np.reshape(inputs[:,:,M//2:M], (-1, M-M//2))
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]

    r_f = 200
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    x_train_part = f_hat.T.astype(np.float32)
else:
    
    print("must compute input PCA")

del train_inputs
del inputs
del Ui, Vi, Uf, f_hat



x_train = np.zeros((M//2 * N, r_f + 1), dtype = np.float32)
y_train = np.zeros(M//2 * N, dtype = np.float32)

for i in range(M//2):
    d_range = range(i*N, (i + 1)*N)
    x_train[d_range , 0:r_f]   = x_train_part[i, :]
    x_train[d_range , r_f]     = xgrid
    y_train[d_range] = outputs[:, i]
    


print("Input dim : ", r_f+1, " output dim : ", 1)
 


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).unsqueeze(-1)

x_normalizer = UnitGaussianNormalizer(x_train)
x_normalizer.encode_(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)
y_normalizer.encode_(y_train)

print("Start training ")
################################################################
# training and evaluation
################################################################

batch_size = 1024 #4096#1024# 512#8192

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)


learning_rate = 0.001

epochs = 200
step_size = 100
gamma = 0.5

layers = 4
model = FNN(r_f + 1, 1, layers, N_neurons) 
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

    torch.save(model, "PARANet_"+str(N_neurons)+"Nd_"+str(ntrain)+".model")
    scheduler.step()

    train_l2/= ntrain

    t2 = default_timer()
    print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)



print("Total time is :", default_timer() - t0, "Total epoch is ", epochs)
