import sys
import numpy as np
sys.path.append('../../../nn')

# import mynn
from mynn import *

N_theta = 100

prefix = "../"
theta = np.load(prefix+"NS_theta_" + str(N_theta) + ".npy")   
K = np.load(prefix+"NS_omega_" + str(N_theta) + ".npy")
        
theta_train, K_train = theta[0::2, :], K[:, :,  0::2]
theta_test, K_test = theta[1::2, :], K[:, :,  1::2]

# Scale the data output
coeff_scale = 1.0
# load data 

N_data, N_theta  =  theta_train.shape
N_x, N_y, N_data = K_train.shape

N_tot = N_x*N_y

L = 2*np.pi
assert(N_x == N_y)
dx = L/(N_x - 1)

input_train  = np.zeros((N_data * N_tot, (N_theta + 2)), dtype=np.float32) # theta, x, y
output_train = np.zeros((N_data * N_tot), dtype=np.float32)

xx = np.linspace(0, L, N_x)
Y, X = np.meshgrid(xx, xx)

# test
i = 20
j = 40
assert(X[i, j] == i*dx and Y[i, j] == j*dx)

X_tot = X.reshape(N_tot)
Y_tot = Y.reshape(N_tot)

for i in range(N_data):
    d_range = range(i*N_tot, (i + 1)*N_tot)
    input_train[d_range , 0:N_theta]     = theta_train[i, :]
    input_train[d_range ,   N_theta]     = X_tot
    input_train[d_range ,   N_theta + 1] = Y_tot 
    output_train[d_range]            = K_train[:, :, i].reshape(N_tot) * coeff_scale
    



x_train = torch.from_numpy(input_train) 
y_train = torch.from_numpy(output_train).unsqueeze(-1)

ds = DirectData(X=x_train, y=y_train)
ds = DataLoader(ds, batch_size=256, shuffle=True)


# TODO change to DirectNet_50
N_neurons = 20

if N_neurons == 20:
    DirectNet = DirectNet_20
elif N_neurons == 50:
    DirectNet = DirectNet_50


model = DirectNet(N_theta+2, 1)
# model = torch.load("DirectNet_"+str(N_neurons)+".model")


# training with adam
loss_fn = torch.nn.MSELoss(reduction='sum')


learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

n_epochs = 0
for epoch in range(n_epochs):
    
    for ix, (_x, _y) in enumerate(ds):
    
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(_x)

        # Compute and print loss.
        loss = loss_fn(y_pred, _y)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
    if epoch % 10 == 0:
        print("[{}/{}], loss: {}".format(epoch, n_epochs, np.round(loss.item(), 3)))
        torch.save(model, "DirectNet_"+str(N_neurons)+".model")

	
# save the model
torch.save(model, "DirectNet_"+str(N_neurons)+".model")


######################################################
N_data, N_theta  =  theta_train.shape
N_x, N_y, N_data = K_train.shape
N_tot = N_x*N_y

output_train = model(torch.from_numpy(input_train)).detach().numpy()
K_pred = np.zeros((N_x, N_y, N_data))

for i in range(N_data):
    d_range = range(i*N_tot, (i + 1)*N_tot)
    K_pred[:,:,i] = output_train[d_range].reshape(N_x, N_y)/coeff_scale


train_ids = np.arange(0, N_data)
errors = np.zeros(N_data)
for train_id in train_ids:

    errors[train_id] =  np.linalg.norm(K_pred[:, :, train_id] - K_train[:, :, train_id])/np.linalg.norm(K_train[:, :, train_id])
    # print(prefix+"data %i, relative error is %.5f:" % (test_id, errors[test_id]))

    if train_id %249 == 0:
        vmin, vmax = None, None
        fig = plt.figure()
        plt.pcolormesh(X, Y, K_train[:, :, train_id], shading="gouraud", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title("Truth")
        fig.savefig(prefix+"Truth_%04i.png" % train_id)

        fig = plt.figure()
        plt.pcolormesh(X, Y, K_pred[:, :, train_id], shading="gouraud", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title("Prediction")
        fig.savefig(prefix+"Prediction_%04i.png" % train_id)
        
        fig = plt.figure()
        plt.pcolormesh(X, Y, K_pred[:, :, train_id] - K_train[:, :, train_id], shading="gouraud")
        plt.colorbar()
        plt.title("Error")
        
        

print("Average training error is ", np.average(errors))
fig = plt.figure()
plt.plot(errors)


######################################################
N_data, N_theta  =  theta_test.shape
N_x, N_y, N_data = K_test.shape

N_data_test = len(theta_test)
input_test = np.zeros((N_data_test*N_tot, (N_theta + 2)), dtype=np.float32) # theta, x, y

for i in range(N_data_test):
    d_range = range(i*N_tot, (i + 1)*N_tot)
    input_test[d_range , 0:N_theta]   = theta_test[i,:]
    input_test[d_range , N_theta]     = X_tot
    input_test[d_range , N_theta + 1] = Y_tot
    


output_test = model(torch.from_numpy(input_test)).detach().numpy()
K_pred = np.zeros((N_x, N_y, N_data))

for i in range(N_data):
    d_range = range(i*N_tot, (i + 1)*N_tot)
    K_pred[:,:,i] = output_test[d_range].reshape(N_x, N_y)/coeff_scale


    

test_ids = np.arange(0, N_data)
errors = np.zeros(N_data)
for test_id in test_ids:

    errors[test_id] =  np.linalg.norm(K_pred[:, :, test_id] - K_test[:, :, test_id])/np.linalg.norm(K_test[:, :, test_id])
    # print(prefix+"data %i, relative error is %.5f:" % (test_id, errors[test_id]))

    if test_id %249 == 0:
        vmin, vmax = None, None
        fig = plt.figure()
        plt.pcolormesh(X, Y, K_test[:, :, test_id], shading="gouraud", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title("Truth")
        fig.savefig(prefix+"Truth_%04i.png" % test_id)

        fig = plt.figure()
        plt.pcolormesh(X, Y, K_pred[:, :, test_id], shading="gouraud", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title("Prediction")
        fig.savefig(prefix+"Prediction_%04i.png" % test_id)
        
        fig = plt.figure()
        plt.pcolormesh(X, Y, K_pred[:, :, test_id] - K_test[:, :, test_id], shading="gouraud")
        plt.colorbar()
        plt.title("Error")
        
        

print("Average test error is ", np.average(errors))
fig = plt.figure()
plt.plot(errors)



