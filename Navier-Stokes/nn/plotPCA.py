import sys
import numpy as np
sys.path.append('../../../nn')
from datetime import datetime

import matplotlib as mpl 
from matplotlib.lines import Line2D 
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
plt.rc("text", usetex=True)         # Crisp axis ticks
plt.rc("font", family="serif")      # Crisp axis labels
# plt.rc("legend", edgecolor="none")  # No boxes around legends

plt.rc("figure",facecolor="#ffffff")
plt.rc("axes",facecolor="#ffffff",edgecolor="#808080",labelcolor="#000000")
plt.rc("savefig",facecolor="#ffffff")
plt.rc("text",color="#000000")
plt.rc("xtick",color="#808080")
plt.rc("ytick",color="#808080")

M = 20000

N = 64

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]

recomputePCA = False
if recomputePCA:
    ntrain = M//2
    N_theta = 100
    prefix = "../../data/"
    # theta = np.load(prefix+"Random_NS_theta_" + str(N_theta) + ".npy")   
    outputs = np.load(prefix+"Random_NS_omega_" + str(N_theta) + ".npy")
    inputs = np.load(prefix+"Random_NS_curl_f_" + str(N_theta) + ".npy")

    acc=0.999


    train_outputs = np.reshape(outputs[:,:,:M//2], (-1, M//2))
    test_outputs  = np.reshape(outputs[:,:,M//2:M], (-1, M-M//2))
    Uo,So,Vo = np.linalg.svd(train_outputs)
    en_g = 1 - np.cumsum(So)/np.sum(So)
    r_g = np.argwhere(en_g<(1-acc))[0,0]
    Ug = Uo[:,:r_g]
    np.save("PCA/NSpca.npy",Ug[:,:4])
else:
    Ug = np.load("PCA/NSpca.npy")


DeepONetPCA_data = np.load("DeepONet/DeepONetPCA_data.npy")
xgrid = np.linspace(0,1,N)
Y, X = np.meshgrid(xgrid, xgrid)

fig = plt.figure(figsize=(6.5,5.5))
subfigs = fig.subfigures(nrows=3,ncols=1)
axs = []
for row,subfig in enumerate(subfigs):
    axs.append(subfig.subplots(1,4))
ims = []
for i in range(4):
    ims.append(axs[0][i].pcolormesh(X,Y,np.reshape(Ug[:, i], (N,N)),shading="gouraud",cmap="gray",vmin=-0.03,vmax=0.03))
    ims.append(axs[1][i].pcolormesh(X, Y, DeepONetPCA_data[:,:,2*i], shading="gouraud",cmap="gray",vmin=0,vmax=0.35))
    ims.append(axs[2][i].pcolormesh(X,Y,DeepONetPCA_data[:,:,2*i+1],shading="gouraud",cmap="gray",vmin=-0.05,vmax=0.05))

    for j in range(3):
        axs[j][i].set_aspect("equal","box")
        axs[j][i].set_axis_off()
    
subfigs[0].suptitle("Leading PCA basis functions",fontsize=16,y=0.95)
subfigs[1].suptitle("Trained DeepONet trunk functions",fontsize=16,y=0.95)
subfigs[2].suptitle("Leading PCA modes of trained DeepONet trunk functions",fontsize=16,y=0.95)

plt.subplots_adjust(left=0.02,right=0.87,bottom=0.02,top=0.90)
cax = []
for i in range(3):
    temp = axs[i][3].get_position()
    cax.append(subfigs[i].add_axes([0.89,temp.y0,0.02,temp.y1-temp.y0]))
    cb = plt.colorbar(ims[i],cax=cax[i])
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_tick_params(width=0.3)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)


fig.savefig("NS-pca-vis.pdf")
plt.close("all")
# plt.colorbar()
