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
plt.rc("legend", edgecolor='none')  # No boxes around legends

plt.rc("figure",facecolor="#ffffff")
plt.rc("axes",facecolor="#ffffff",edgecolor="#808080",labelcolor="#000000")
plt.rc("savefig",facecolor="#ffffff")
plt.rc("text",color="#000000")
plt.rc("xtick",color="#808080")
plt.rc("ytick",color="#808080")

DeepONetPCA_data = np.load("DeepONet/DeepONetPCA_data.npy")
M = 20000

N = 200
ntrain = M//2
N_theta = 100
prefix = "../../data/"  
a0 = np.load(prefix+"adv_a0.npy")
aT = np.load(prefix+"adv_aT.npy")


acc = 0.999

xgrid = np.linspace(0,1,N)
dx    = xgrid[1] - xgrid[0]

inputs  = a0
outputs = aT

computePCA = False
if computePCA:
    train_outputs = outputs[:,:M//2] 
    Uo,So,Vo = np.linalg.svd(train_outputs)
    en_g = 1 - np.cumsum(So)/np.sum(So)
    r_g = np.argwhere(en_g<(1-acc))[0,0]
    Ug = Uo[:,:r_g]
    np.save("PCA/PCA.npy",Ug)
else:
    Ug = np.load("PCA/PCA.npy")
    
fig = plt.figure(figsize=(6.5,4))
subfigs = fig.subfigures(nrows=3,ncols=1)
axs = []
for row,subfig in enumerate(subfigs):
    axs.append(subfig.subplots(1,4,sharey=True))
ims = []
for i in range(4):
    ims.append(axs[0][i].plot(xgrid,Ug[:, i],color="#262936"))
    ims.append(axs[1][i].plot(xgrid,DeepONetPCA_data[:,2*i],color="#262936"))
    ims.append(axs[2][i].plot(xgrid,DeepONetPCA_data[:,2*i+1],color="#262936",clip_on=False))

    axs[0][i].set_xticklabels([])
    # axs[0][i].set_ylim([-0.12,0.12])
    axs[1][i].set_xticklabels([])
    # axs[1][i].autoscale()

    # axs[2][i].set_ylim([-0.25,0.25])
    # if i>0:
    #     axs[2][i].set_yticklabels([])
    #     axs[0][i].set_yticklabels([])


    for j in range(3):
        axs[j][i].spines["right"].set_visible(False)
        axs[j][i].spines["top"].set_visible(False)
    
subfigs[0].suptitle("Leading PCA basis functions",fontsize=16,y=1)
subfigs[1].suptitle("Trained DeepONet trunk functions",fontsize=16,y=1)
subfigs[2].suptitle("Leading PCA modes of trained DeepONet trunk functions",fontsize=16,y=1)

plt.subplots_adjust(left=0.06,right=0.98,bottom=0.15,top=0.8,hspace=0.5)
fig.savefig("advection-pca-vis.pdf")
plt.close("all")

