import numpy as np
import matplotlib.pyplot as plt
plt.rc("figure", dpi=300)           # High-quality figure ("dots-per-inch")
plt.rc("text", usetex=True)         # Crisp axis ticks
plt.rc("font", family="serif")      # Crisp axis labels
plt.rc("legend", edgecolor="none")  # No boxes around legends
plt.rc("figure",facecolor="#ffffff")
plt.rc("axes",facecolor="#ffffff",edgecolor="#808080",labelcolor="#000000")
plt.rc("savefig",facecolor="#ffffff")
plt.rc("text",color="#000000")
plt.rc("xtick",color="#808080")
plt.rc("ytick",color="#808080")
colors = ["#3A637B", "#C4A46B", "#FF6917", "#D44141" ] 
markers = ["o", "s", "^", "*"]
linestyle = ["dotted", "-.", "--", "-", ]

lineskip = [2, 2, 2, 2]
lineend = [999,999,199,499]

nn_names = ["PCA", "DeepONet", "PARA", "FNO"]
N2_vals = [312,624,1250,2500,5000,10000,20000,40000]
sizes = [16,64,128,256,512]
fno_sizes = [2,4,8,16,32]

data = np.zeros((1000,8,5,4))
for idnn,nn in enumerate(nn_names):
    for ids, sz in enumerate(sizes):
        if nn == "FNO":
            sz = fno_sizes[ids]

        for idN, N2 in enumerate(N2_vals):
            if idnn == 2 and N2 == 40000:
                continue
            filename = nn+"/output/"+str(N2)+"-"+str(sz)+"-train.out"
            fo = open( filename) #open your file for reading ('r' by default)
            ll = 0
            for line in fo: # parse the file line by line
                if ll >= lineskip[idnn] and ll <= lineskip[idnn]+lineend[idnn]:
                    temp = line.split()
                    data[ll-lineskip[idnn],idN,ids,idnn] = float(temp[-1])
                    # print(temp[-1])
                ll += 1
            fo.close()

fig,ax = plt.subplots(5,4,sharex="col", figsize=(8,7))
for idnn,nn in enumerate(nn_names):
    for ids, sz in enumerate(sizes):
        for idN, N2 in enumerate(N2_vals[4:]):
            ax[ids,idnn].semilogy(data[:,idN,ids,idnn],color=colors[idnn],linestyle=linestyle[idN],label="$N = "+str(N2/2)+"$")
        ax[ids,idnn].set_xlim([0,lineend[idnn]])
        if ids == 0:
            ax[ids,idnn].set_title(nn,pad=20,fontsize=14)
        if idnn == 0:
            ax[ids,idnn].set_ylabel("$w = "+str(sz)+ " / d_f = "+str(fno_sizes[ids])+"$")
        if ids == 4:
            ax[ids,idnn].set_xlabel("Epoch")
handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.99),ncol=4,frameon=False,fontsize=12,labelcolor="linecolor")
plt.subplots_adjust(left=0.08,right=0.98,bottom=0.07,top=0.94,hspace=0.2,wspace=0.38)
fig.savefig("HH-trainloss-v-epoch.png")
plt.close("all")
