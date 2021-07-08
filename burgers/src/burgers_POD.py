import numpy as np 
import matplotlib as mpl 
from matplotlib.lines import Line2D 
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sp

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

def get_x_sq(X):
    if X.ndim==1:
        w = len(X)
        X2 = X[0]*X

        for j in range(1,w):
            X2 = np.concatenate((X2,X[j]*X[j:]))

    elif X.ndim==2:
        K,w = X.shape
        X2 = X[:,0][:,np.newaxis]*X
        for j in range(1,w):
            temp = X[:,j][:,np.newaxis]*X[:,j:]
            X2 = np.concatenate((X2,temp),axis=1)

    return X2

def getF(N,dx):
    ii = np.repeat(np.arange(N),2)

    # column index of the squares
    tt = np.cumsum(np.arange(N,0,-1))
    tt[1:] = tt[:-1]
    tt[0] = 0

    jj = np.zeros((2*N,))
    jj[1::2] = np.roll(tt,1) # from the left
    jj[::2] = np.roll(tt,-1) # from the right

    vr = [-1., 1.]/(2*dx) # 1/2 is for central diff
    vv = np.tile(vr,N)
    F = sp.coo_matrix((vv,(ii,jj)),shape=(N,N*(N+1)/2))
    return F

def getH(N,dx):
    ii = np.repeat(np.arange(N),2)
    tt = np.arange(N)*N + np.arange(N)
    jj = np.zeros((2*N,))
    jj[1::2] = np.roll(tt,1) # from the left
    jj[::2] = np.roll(tt,-1) # from the right
    vr = [-1., 1.]/(2*dx) # 1/2 is for central diff
    vv = np.tile(vr,N)
    H = sp.coo_matrix((vv,(ii,jj)),shape=(N,N**2))
    return H

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

def semiimplict(u0,t,A,H):
    K = t.size-1
    dt = t[1] - t[0]
    N = u0.size
    u_traj = np.zeros((N,K+1))
    ImdtA = np.eye(N) -dt*A

    u_traj[:,0] = u0;
    for k in range(K):
        u = u_traj[:,k]
        du2dx = H.dot(np.kron(u,u))
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

states = np.load('../data/june18_states.npz')

init = states['init']
t05 = states['t05']
t1 = states['t1']

bases = np.load('../data/june18_bases.npz')
s0 = bases['s0']
en0 = 1 - np.cumsum(s0**2)/np.sum(s0**2)
s05 = bases['s5']
en05 = 1 - np.cumsum(s05**2)/np.sum(s05**2)
s1 = bases['s1']
en1 = 1 - np.cumsum(s1**2)/np.sum(s1**2)

acc = 0.999
r0 = np.argwhere(en0<(1-acc))[0,0]

A = getA(N-1,dx,nu,bcs)
H = getH(N,dx)

errs = np.zeros((4,2))
i = 0;
# for r in range(5,21,5):
#     Ur = bases['U0'][:,:r]
#     Ahat = np.dot(Ur.T,np.dot(A,Ur))

#     temp = H.dot(np.kron(Ur,Ur))
#     Hhat = np.dot(Ur.T,temp)

#     uhat05 = np.zeros((r,3*M))
#     uhat1 = np.zeros((r,3*M))
#     for j in range(3*M):
#         uhat0 = np.dot(Ur.T,init[:,j])
#         uhat = semiimplict(uhat0,t,Ahat,Hhat)
#         uhat05[:,j] = uhat[:,201]
#         uhat1[:,j] = uhat[:,401]

#     truth05 = np.dot(Ur.T,t05)
#     truth1 = np.dot(Ur.T,t1)

#     errs[i,0] = np.sum((truth05-uhat05)**2)/np.sum(truth05**2)
#     errs[i,1] = np.sum((truth1 - uhat1)**2)/np.sum(truth1**2)

#     print errs
#     i +=1 

temp = np.load('june18_romerrs.npz')
errs = temp['errs']

r_vals = np.arange(5,21,5)

fig,ax = plt.subplots(figsize=(3,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(r_vals,errs[:,0], lw=0.5, color=color1,label='t = 0.5')
ax.semilogy(r_vals,errs[:,1], lw=0.5, color=color2,label='t = 1')
ax.legend()
plt.xlabel('$r$')
plt.title('error in POD coefficient prediction')
plt.savefig('../tfigs/POD_err.png',pad_inches=3)
plt.close()


## verify F matrix works 
# u = init[:,0]
# u2 = u**2
# du2dx = (np.roll(u2,1) - np.roll(u2,-1))/(2*dx)

# F = getF(N,dx)
# usq = get_x_sq(u)
# temp = F.dot(usq)

# H = getH(N,dx)
# temp2 = H.dot(np.kron(u,u))

# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.plot(xgrid,du2dx, lw=0.5, color=color1,label='diff')
# ax.plot(xgrid,temp,'+', lw=0.5, color=color2,label='F*u2')
# ax.plot(xgrid,temp2,'o', lw=0.5, color=color3,label='H*uxu')
# ax.legend()
# plt.xlabel('$x$')
# plt.savefig('../tfigs/F_test.png',pad_inches=3)
# plt.close()

## PLOT SVD DECAY
# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.semilogy(en0, lw=0.5, color=color1,label='t=0')
# ax.semilogy(en05, lw=0.5, color=color2,label='t=0.5')
# ax.semilogy(en1, lw=0.5, color=color3,label='t=1')
# ax.legend()
# plt.xlabel('$r$')
# plt.title('energy lost')
# plt.savefig('../tfigs/svd.png',pad_inches=3)
# plt.close()

# # plot POD modes at t = 0, 0.5, 1
# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.plot(xgrid,bases['U0'][:,:5])
# plt.xlabel('$x$')
# plt.title('leading POD modes at t = 0')
# plt.savefig('../tfigs/pod_init.png',pad_inches=3)
# plt.close()

# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.plot(xgrid,bases['U5'][:,:5])
# plt.xlabel('$x$')
# plt.title('leading POD modes at t = 0.5')
# plt.savefig('../tfigs/pod_half.png',pad_inches=3)
# plt.close()

# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.plot(xgrid,bases['U1'][:,:5])
# plt.xlabel('$x$')
# plt.title('leading POD modes at t = 1')
# plt.savefig('../tfigs/pod_1.png',pad_inches=3)
# plt.close()

# # plot state for first five realizations
# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.plot(xgrid,init[:,:5])
# plt.xlabel('$x$')
# plt.title('realizations t = 0')
# plt.savefig('../tfigs/state_init.png',pad_inches=3)
# plt.close()

# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.plot(xgrid,t05[:,:5])
# plt.xlabel('$x$')
# plt.title('realizations t = 0.5')
# plt.savefig('../tfigs/state_half.png',pad_inches=3)
# plt.close()

# fig,ax = plt.subplots(figsize=(3,3))
# fig.subplots_adjust(bottom=0.2,left = 0.15)
# ax.plot(xgrid,t1[:,:5])
# plt.xlabel('$x$')
# plt.title('realizations t = 1')
# plt.savefig('../tfigs/state_1.png',pad_inches=3)
# plt.close()