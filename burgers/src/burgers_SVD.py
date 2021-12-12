import numpy as np 
import matplotlib as mpl 
from matplotlib.lines import Line2D 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as sla
from GRF1D import *

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

def getA(N,dx,nu,bcs):
    e = np.ones(N+1)
    A = nu/dx**2*(np.diag(-2*e) + np.diag(e[1:],1) + np.diag(e[1:],-1))

    if bcs == 'DD':
        A[0,:2]   = [1, 0]; 
        A[N,N-1:] = [0, 1];
    elif bcs == 'periodic':
        A[0,N] = nu/dx**2
        A[N,0] = nu/dx**2
    else:
        print("undefined boundary condition in getA")

    return A


def run_burgers(u0,t,dx,nu,bcs):
    K = t.size-1
    dt = t[1] - t[0]
    N = u0.size-1
    u_traj = np.zeros((N+1,K+1))


    A = getA(N,dx,nu,bcs)
    ImdtA = np.eye(N+1) -dt*A

    u_traj[:,0] = u0;
    for k in range(K):
        u = u_traj[:,k]
        u2 = u**2

        if bcs == 'DD':
            du2dx = (u2[2:] - u2[:-2])/(2*dx)
            u[0]  = 0
            u[-1] = 0
            u[1:-1] -= dt*0.5*du2dx
            u_traj[:,k+1] = np.linalg.solve(ImdtA,u)
        elif bcs == 'periodic':
            du2dx = (np.roll(u2,1) - np.roll(u2,-1))/(2*dx)
            u_traj[:,k+1] = np.linalg.solve(ImdtA,u-dt*0.5*du2dx)

    return u_traj

##########################################
# Tests start here
##########################################

# spatial domain setup
d     = 1
N     = 128
xgrid = np.linspace(0,1,N+1)
xgrid = xgrid[:-1]
dx    = xgrid[1] - xgrid[0]
bcs   = 'periodic'

# time domain setup
T     = 2.
K     = 800
t     = np.linspace(0,T,K+1)

# burgers param setup
nu    = 0.01
tau   = 7.
alpha = 2.5

grf = GaussianRandomField1D(tau, alpha, bc=2)

M = 2048

Ks = 100

for j in range(3):
    dat = np.load('../data/T'+str(int(T))+'_N'+str(N)+'_K'+str(K)+'_M'+str(M)+'_traj'+str(j)+'.npz')
    if j == 0:
        traj = dat['traj'][:,range(0,K/2+1,K/Ks),:]
    else:
        traj = np.concatenate((traj,dat['traj'][:,range(0,K/2+1,K/Ks),:]),axis=2)

# SVD all downsampled snaps up to t = 1
# Ua,sa,Va = np.linalg.svd(traj.reshape((N,(Ks/2+1)*3*M)))

# SVD all downsampled snaps up to t = 0.5
# Ub,sb,Vb = np.linalg.svd(traj[:,:(Ks/4+1),:].reshape((N,(Ks/4+1)*3*M)))

U0,s0,V0 = np.linalg.svd(traj[:,0,:].reshape((N,3*M)))

U5,s5,V5 = np.linalg.svd(traj[:,Ks/4+1,:].reshape((N,3*M)))

U1,s1,V1 = np.linalg.svd(traj[:,-1,:].reshape((N,3*M)))

np.savez('../data/june18_bases.npz',U0=U0,s0=s0,U5=U5,s5=s5,U1=U1,s1=s1)

init = traj[:,0,:].reshape((N,3*M))
t05 = traj[:,Ks/4+1,:].reshape((N,3*M))
t1 = traj[:,-1,:].reshape((N,3*M))
np.savez('../data/june18_states.npz',init=init,t05 = t05, t1=t1)
# dat0 = dat0['traj'].reshape((N,(K+1)*M))
# dat1 = dat1['traj'].reshape((N,(K+1)*M))
# dat2 = dat2['traj'].reshape((N,(K+1)*M))