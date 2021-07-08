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

def colnorm(X):
    return np.sqrt(np.sum(X**2,axis=0))

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

A = getA(N-1,dx,nu,bcs)
H = getH(N,dx)

grf = GaussianRandomField1D(tau, alpha, bc=2)

M = 2048

Ks = 100

##################################
# this loads all the downsampled
##################################
for j in range(3):
    dat = np.load('../data/T'+str(int(T))+'_N'+str(N)+'_K'+str(K)+'_M'+str(M)+'_traj'+str(j)+'.npz')
    if j == 0:
        traj = dat['traj'][:,range(0,K/2+1,K/Ks),:]
    else:
        traj = np.concatenate((traj,dat['traj'][:,range(0,K/2+1,K/Ks),:]),axis=2)

# want to do PCA just on training inputs (first 1024 inputs)
n_train = 1024
n_test = 3*M-n_train
f_train = traj[:,0,:n_train]
f_test  = traj[:,0,n_train:]
Uf,sf,Vf = np.linalg.svd(f_train)
enf = np.sqrt(1 - np.cumsum(sf**2)/np.sum(sf**2))

# test at t = 0.5
g_train05 = traj[:,25,:n_train]
g_test05  = traj[:,25,n_train:]

# test at t = 1
g_train1 = traj[:,-1,:n_train]
g_test1 = traj[:,-1,n_train:]

# preallocate space for r-tests
r_vals = np.arange(5,51,5)
pef = np.zeros((len(r_vals),4))
peg1 = np.zeros((len(r_vals),2))
peg05 = np.zeros((len(r_vals),2))

# dividing norms
ftrainF = np.linalg.norm(f_train,'fro')
ftraincol = colnorm(f_train)
ftestF = np.linalg.norm(f_test,'fro')
ftestcol = colnorm(f_test)

gtraincol1 = colnorm(g_train1)
gtestcol1 = colnorm(g_test1)
gtraincol05 = colnorm(g_train05)
gtestcol05 = colnorm(g_test05)

errs1 = np.zeros((len(r_vals),4))
errs05 = np.zeros((len(r_vals),4))
for rr in range(len(r_vals)):
    r = r_vals[rr]
    Ur = Uf[:,:r]
    Pr = np.matmul(Uf[:,:r],Uf[:,:r].T)
    pf_train = np.matmul(Pr,f_train)
    pef_train = pf_train - f_train
    pef[rr,2] = np.sum(colnorm(pef_train)/ftraincol)/n_train
    pef[rr,0] = np.linalg.norm(pef_train,'fro')/ftrainF

    pf_test = np.matmul(Pr,f_test)
    pef_test = pf_test - f_test 
    pef[rr,3] = np.sum(colnorm(pef_test)/ftestcol)/n_test
    pef[rr,1] = np.linalg.norm(pef_test,'fro')/ftestF

    peg1_train = np.matmul(Pr,g_train1) - g_train1
    peg1[rr,0] = np.sum(colnorm(peg1_train)/gtraincol1)/n_train
    peg1_test = np.matmul(Pr,g_test1) - g_test1
    peg1[rr,1] = np.sum(colnorm(peg1_test)/gtestcol1)/n_test

    peg05_train = np.matmul(Pr,g_train05) - g_train05
    peg05[rr,0] = np.sum(colnorm(peg05_train)/gtraincol05)/n_train
    peg05_test = np.matmul(Pr,g_test05) - g_test05
    peg05[rr,1] = np.sum(colnorm(peg05_test)/gtestcol05)/n_test

    Ahat = np.dot(Ur.T,np.dot(A,Ur))
    temp = H.dot(np.kron(Ur,Ur))
    Hhat = np.dot(Ur.T,temp)

    uhat05 = np.zeros((r,n_train))
    uhat1 = np.zeros((r,n_train))
    for j in range(n_train):
        uhat0 = np.dot(Ur.T,f_train[:,j])
        uhat = semiimplict(uhat0,t[:401],Ahat,Hhat)
        uhat05[:,j] = uhat[:,200]
        uhat1[:,j] = uhat[:,-1]

    truth05 = np.dot(Ur.T,g_train05)
    truth1 = np.dot(Ur.T,g_train1)

    errs05[rr,0] = np.sum(colnorm(truth05-uhat05)/colnorm(truth05))/n_train
    errs05[rr,1] = np.sum(colnorm(np.dot(Ur,uhat05)-g_train05)/gtraincol05)/n_train
    errs1[rr,0] = np.sum(colnorm(truth1 - uhat1)/colnorm(truth1))/n_train
    errs1[rr,1] = np.sum(colnorm(np.dot(Ur,uhat1)-g_train1)/gtraincol1)/n_train

    uhat05 = np.zeros((r,n_test))
    uhat1 = np.zeros((r,n_test))
    for j in range(n_test):
        uhat0 = np.dot(Ur.T,f_test[:,j])
        uhat = semiimplict(uhat0,t[:401],Ahat,Hhat)
        uhat05[:,j] = uhat[:,200]
        uhat1[:,j] = uhat[:,-1]

    truth05 = np.dot(Ur.T,g_test05)
    truth1 = np.dot(Ur.T,g_test1)

    errs05[rr,2] = np.sum(colnorm(truth05-uhat05)/colnorm(truth05))/n_test
    errs05[rr,3] = np.sum(colnorm(np.dot(Ur,uhat05)-g_test05)/gtestcol05)/n_test
    errs1[rr,2] = np.sum(colnorm(truth1 - uhat1)/colnorm(truth1))/n_test
    errs1[rr,3] = np.sum(colnorm(np.dot(Ur,uhat1)-g_test1)/gtestcol1)/n_test

    print r
    print errs05[rr,:]
    print errs1[rr,:]

# # plot SVD of f, projection errors in Frobenius norm
fig,ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(enf, lw=0.5, color=color1,label='$\sqrt{\sum_{i=r+1}\sigma_i^2\slash\sum_{i=1}\sigma_i^2}$')
ax.semilogy(r_vals,pef[:,0], lw=0.5, color=color2,label='training: $||F - P_rF||_F\slash||F||_F$')
ax.semilogy(r_vals,pef[:,1], lw=0.5, color=color3,label='test: $||F - P_rF||_F\slash||F||_F$')
ax.legend()
plt.xlabel('$r$')
plt.title('SVD on inputs $f = u_0$')
plt.savefig('../tfigs/bm_svd_f.png',pad_inches=3)
plt.close()

# # plot mean projection errors over INPUT data
fig,ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(r_vals,pef[:,2], lw=2, color=color2,label='training: $||f - P_rf||_2/||f||_2$')
ax.semilogy(r_vals,pef[:,3],':', lw=2, color=color3,label='test: $||f - P_rf||_2/||f||_2$')
ax.legend()
plt.xlabel('$r$')
plt.title('Mean relative projection error on inputs $f = u_0$')
plt.savefig('../tfigs/bm_avgprojerr_f.png',pad_inches=3)
plt.close()

# # plot mean projection errors over OUTPUT data t = 1
fig,ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(r_vals,peg1[:,0], lw=2, color=color2,label='training: $||g - P_rg||_2/||g||_2$')
ax.semilogy(r_vals,peg1[:,1],':', lw=2, color=color3,label='test: $||g - P_rg||_2/||g||_2$')
ax.legend()
plt.xlabel('$r$')
plt.title('Mean rel proj error on outputs $g = u(t = 1)$')
plt.savefig('../tfigs/bm_avgprojerr_g1.png',pad_inches=3)
plt.close()

# # plot mean proj err over OUTPUT data t = 0.5
fig,ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(r_vals,peg05[:,0], lw=2, color=color2,label='training: $||g - P_rg||_2/||g||_2$')
ax.semilogy(r_vals,peg05[:,1],':', lw=2, color=color3,label='test: $||g - P_rg||_2/||g||_2$')
ax.legend()
plt.xlabel('$r$')
plt.title('Mean rel proj error on outputs $g = u(t = 0.5)$')
plt.savefig('../tfigs/bm_avgprojerr_g05.png',pad_inches=3)
plt.close()

#plot mean pred err over OUTPUT data t = 0.5
fig,ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(r_vals,errs05[:,0], lw=2, color=color2,label='training coeffs')
ax.semilogy(r_vals,errs05[:,1],':', lw=2, color=color2,label='training full state')
ax.semilogy(r_vals,errs05[:,2], lw=2, color=color3,label='test coeffs')
ax.semilogy(r_vals,errs05[:,3],':', lw=2, color=color3,label='test full state')
ax.legend()
plt.xlabel('$r$')
plt.title('Mean rel prediction u(t = 0.5)')
plt.savefig('../tfigs/bm_pred_g05.png',pad_inches=3)
plt.close()

#plot mean pred err over OUTPUT data t = 1
fig,ax = plt.subplots(figsize=(4,3))
fig.subplots_adjust(bottom=0.2,left = 0.15)
ax.semilogy(r_vals,errs1[:,0], lw=2, color=color2,label='training coeffs')
ax.semilogy(r_vals,errs1[:,1],':', lw=2, color=color2,label='training full state')
ax.semilogy(r_vals,errs1[:,2], lw=2, color=color3,label='test coeffs')
ax.semilogy(r_vals,errs1[:,3],':', lw=2, color=color3,label='test full state')
ax.legend()
plt.xlabel('$r$')
plt.title('Mean rel prediction u(t = 1)')
plt.savefig('../tfigs/bm_pred_g1.png',pad_inches=3)
plt.close()