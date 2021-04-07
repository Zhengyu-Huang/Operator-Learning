import numpy as np
from mynn import *

prefix = "Random_Direct_"
L = 1.0
Nx = Ny = 101   #number of points
input_test, output_test = preprocess_data([86, 91])
κs_half_test = np.transpose(output_test.reshape(-1, (Nx*(Ny + 1)//2)))
N_data = κs_half_test.shape[1]
κs_test = np.zeros((Nx, Ny, N_data))
# load the model
model = torch.load("DirectKernelNet.model")
  
output_pred = model(torch.from_numpy(input_test)).detach().numpy()
κs_half_pred = np.transpose(output_pred.reshape(-1, (Nx*(Ny + 1)//2)))
κs_pred = np.zeros((Nx, Ny, N_data))

k = 0
for ix in range(Nx):
    for iy in range(ix, Ny):
        κs_test[ix, iy, :] = κs_test[iy, ix, :] = κs_half_test[k, :]
        κs_pred[ix, iy, :] = κs_pred[ix, iy, :] = κs_half_pred[k, :]
        k += 1


xx = np.linspace(0, L, Nx)
Y, X = np.meshgrid(xx, xx)


test_ids = np.arange(0, N_data)
errors = np.zeros(len(test_ids))
for test_id in test_ids:


    errors[test_id] =  np.linalg.norm(κs_pred[:, :, test_id] - κs_test[:, :, test_id])/np.linalg.norm(κs_test[:, :, test_id])
    print(prefix+"data %i, relative error is %.5f:" % (test_id, errors[test_id]))

    if test_id %100 == 0:
        vmin, vmax = np.min(κs_test[:, :, test_id]), np.max(κs_test[:, :, test_id])
        fig = plt.figure()
        plt.pcolormesh(X, Y, κs_test[:, :, test_id], shading="gouraud", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title("Truth")
        fig.savefig(prefix+"Truth_%i.png" % test_id)

        fig = plt.figure()
        plt.pcolormesh(X, Y, κs_pred[:, :, test_id], shading="gouraud", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title("Prediction")
        fig.savefig(prefix+"Prediction_%i.png" % test_id)

print("Average error is ", np.average(errors))