import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from timeit import default_timer

N_theta = 100
prefix = "/central/scratch/dzhuang/Helmholtz_data/"
sigma = np.load(prefix+"Random_UnitCell_sigma_" + str(N_theta) + ".npy")
Fn = np.load(prefix+"Random_UnitCell_Fn_" + str(N_theta) + ".npy")
XY = np.load(prefix+"Random_UnitCell_XY_" + str(N_theta) + ".npy")

N_data = sigma.shape[1]

N = 21
xgrid = np.linspace(0,1,N)
N_f = 41
xgrid_f = np.linspace(0,1,N_f)
Y, X = np.meshgrid(xgrid_f, xgrid_f)

sigma_f = np.zeros((N_f, N_f, N_data))
Fn_f = np.zeros((N_f, N_f, N_data))





for i in range(N_data):
    
    rbf = interpolate.Rbf(XY[:, 0], XY[:, 1], sigma[:, i])
    sigma_f[:, :, i] = rbf(X, Y)
    
    int_c = interpolate.interp1d(xgrid, Fn[:, i], kind = 'cubic')
    Fn_f[:,:, i] = np.outer(int_c(xgrid_f) , np.ones(N_f))

    
np.save(prefix+"Random_UnitCell_sigma_" + str(N_theta) + "_interp.npy", sigma_f)
np.save(prefix+"Random_UnitCell_Fn_" + str(N_theta) + "_interp.npy", Fn_f)
