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
traj = np.zeros((N,K+1,M))
for m in range(M):
    theta = np.random.standard_normal(N+1)
    # u0 = np.squeeze(np.matmul(LC,theta))
    u0 = grf.draw(theta)
    u0 = u0[:-1]

    u = run_burgers(u0,t,dx,nu,bcs)
    # fig,ax = plt.subplots(figsize=(3,3))
    # fig.subplots_adjust(bottom=0.2,left = 0.15)
    # ax.plot(xgrid,u0, lw=0.5, color=color1)
    # for i in range(10,100,10):
    #     ax.plot(xgrid,u[:,i], lw=0.5, color=(20./255, 148./255, 217./255,float(100-i)/100))
    # plt.xlabel('$x$')
    # plt.savefig('tfigs/burgtraj'+str(m)+'.png',pad_inches=3)
    # plt.close()

    if np.mod(m,50)==0:
        print(str(m))

    # inputs[:,m] = u0
    # outputs[:,m] = u[:,-1]
    traj[:,:,m] = u

np.savez('../data/T'+str(int(T))+'_N'+str(N)+'_K'+str(K)+'_M'+str(M)+'_traj2.npz',traj = traj, t= t)
