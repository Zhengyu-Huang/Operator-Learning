using NPZ
using LinearAlgebra
using PyPlot

# script to plot Helmholtz input-output map and median/maximum error cases
# can choose color option for paper plots or for darkslides, or make your own color option in plotdefaults.jl


coloroption = "darkslides"
include("plotdefaults.jl")
include("Data-NN-Plot.jl") # load data
probs = ["Navier-Stokes","Navier-Stokes", "Helmholtz","Helmholtz", "Structural mechanics","Structural mechanics"]





########### Error vs Data
ylims = [1e-3 10; 1e-3 10; 1e-2 10; 1e-2 10; 0.04 0.5; 0.04 0.5]
ypad = [1, 1, 1.5,1.5, 8,1]

fig, ax = PyPlot.subplots(ncols = 4,nrows=6, sharex=true, figsize=(6.8,9))

N_Data = [156, 312, 625, 1250, 2500, 5000, 10000, 20000]

for i = 1:6 # test problems
    for j = 1:4 # plot columns
        ax[i,j].loglog(N_Data, ylims[i,2]*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",  linewidth=0.5,linestyle="dashed")
        if i == 5 || i == 6
            ax[i,j].text(1250,ylims[i,2]*sqrt(N_Data[1])./sqrt(1250),"1/√N",color="#bababa",fontsize=8)
        else
            ax[i,j].text(5000,ylims[i,2]*sqrt(N_Data[1])./sqrt(5000),"1/√N",color="#bababa",fontsize=8)
        end

        if i==1 || i == 3 || i == 5
            ax[i,j].loglog(N_Data,PCA_Data[j:5:40,5,div(i+1,2)],color=colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none",label="PCA-Net")
            ax[i,j].loglog(N_Data,PCA_Out_Of_Distribution_Data[j:5:40,5,div(i+1,2)],color=colors[1], linestyle="solid", marker = markers[1], fillstyle="none",label="PCA-Net (OOD)")

            ax[i,j].loglog(N_Data,DeepONet_Data[j:5:40,5,div(i+1,2)],color=colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none",label="DeepONet")
            ax[i,j].loglog(N_Data,DeepONet_Out_Of_Distribution_Data[j:5:40,5,div(i+1,2)],color=colors[2], linestyle="solid", marker = markers[2], fillstyle="none",label="DeepONet (OOD)")
        else
            n_ind = (i==6 ? 8 : 7)
            ax[i,j].loglog(N_Data[1:n_ind],PARA_Data[j:5:40,5,div(i+1,2)][1:n_ind],color=colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none",label="PARA-Net")
            ax[i,j].loglog(N_Data[1:n_ind],PARA_Out_Of_Distribution_Data[j:5:40,5,div(i+1,2)][1:n_ind],color=colors[3], linestyle="solid", marker = markers[3], fillstyle="none",label="PARA-Net (OOD)")


            ax[i,j].loglog(N_Data,FNO_Data[j:5:40,5,div(i+1,2)],color=colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none",label="FNO")
            ax[i,j].loglog(N_Data,FNO_Out_Of_Distribution_Data[j:5:40,5,div(i+1,2)],color=colors[4], linestyle="solid", marker = markers[4], fillstyle="none",label="FNO (OOD)")
        end
        
        ax[i,j].set_ylim(ylims[i,:])
        ax[i,j].set_ylim(ylims[i,:])

        # set titles on top row
        if i == 1
            ax[i,j].set_title(sizes[j],pad=40,color=lbl)
        end

        # set xlabels on bottom row
        if i == 6
            ax[i,j].set_xlabel(latexstring("Training data ",L"N"),labelpad=2,color=lbl)
        end

        # gray spines and ticks 
        ax[i,j].spines["top"].set_visible(false)
        ax[i,j].spines["right"].set_visible(false)
        ax[i,j].spines["left"].set_color(tk)
        ax[i,j].spines["left"].set_linewidth(0.3)
        ax[i,j].spines["bottom"].set_color(tk)
        ax[i,j].spines["bottom"].set_linewidth(0.3)
        ax[i,j].set_xticks(N_Data[2:2:end])
        ax[i,j].set_xticklabels(["312","1250","5k","20k"])
        ax[i,j][:xaxis][:set_tick_params](colors=tk,width=0.3)
        ax[i,j][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
        ax[i,j][:yaxis][:set_tick_params](colors=tk,width=0.3)
        ax[i,j][:yaxis][:set_tick_params](which="minor",left=false)
        
        if i == 5 || i == 6
            ax[i,j].set_yticks([0.05, 0.1, 0.2, 0.4])
            ax[i,j][:yaxis][:set_minor_formatter](plt.NullFormatter())
        end

        # set labels on left column and remove tick labels elsewhere
        if j == 1
            ax[i,j].set_ylabel(probs[i],labelpad=ypad[i],color=lbl)
            if i == 5 || i == 6
                ax[i,j].set_yticklabels(["0.05", "0.1", "0.2", "0.5"])
            end
        else
            ax[i,j].set_yticklabels("")
        end
    end
end

handles, labels = ax[1,1].get_legend_handles_labels()
handles_2, labels_2 = ax[2,1].get_legend_handles_labels()
plt.subplots_adjust(bottom=0.05,top=0.92,left=0.13,right=0.98,hspace=0.2)
# plt.tight_layout()
fig.legend([handles; handles_2],[labels;labels_2],loc = "upper center",bbox_to_anchor=(0.5,0.98),ncol=4,frameon=false,fontsize=9,labelcolor="linecolor")
fig.text(0.01, 0.5, "Test Error", ha="left", va="center", rotation="vertical",fontsize=14,color=lbl)
plt.savefig("All-Error-Data-out-of-d-"*coloroption*".pdf")
plt.close("all")

# paper plot simplified
# uses only the w = 256/ df=16 results

ylims = [1e-3 5; 1e-3 10; 1e-2 10; 1e-2 10; 0.04 0.5; 0.04 0.5]
fig, ax = PyPlot.subplots(ncols = 4,nrows=2, sharey="row",sharex=true, figsize=(6.8, 3))

for i = 1:2
    for j = 1:4
        if i == 1
            ax[i,j].loglog(N_Data, ylims[1,2]*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",  linewidth=0.5,linestyle="dashed")
            ax[i,j].text(1250,ylims[1,2]*sqrt(N_Data[1])./sqrt(1250),"1/√N",color="#bababa",fontsize=8)
        elseif i == 2
            ax[i,j].loglog(N_Data, ylims[5,2]*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",  linewidth=0.5,linestyle="dashed")
            ax[i,j].text(312,ylims[5,2]*sqrt(N_Data[1])./sqrt(312),"1/√N",color="#bababa",fontsize=8)
        end
    end
    if i == 1
        ii = 1
    elseif i == 2
        ii = 3
    end
    ax[i,1].loglog(N_Data,PCA_Data[4:5:40,5,ii],color=colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none",label="In distribution")
    ax[i,1].loglog(N_Data,PCA_Out_Of_Distribution_Data[4:5:40,5,ii],color=colors[1], linestyle="solid", marker = markers[1], fillstyle="full",label="Out of distribution")
    ax[i,2].loglog(N_Data,DeepONet_Data[4:5:40,5,ii],color=colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none",label="DeepONet")
    ax[i,2].loglog(N_Data,DeepONet_Out_Of_Distribution_Data[4:5:40,5,ii],color=colors[2], linestyle="solid", marker = markers[2], fillstyle="full",label="DeepONet (OOD)")
    ax[i,3].loglog(N_Data[1:end-1],PARA_Data[4:5:40,5,ii][1:end-1],color=colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none",label="PARA-Net")
    ax[i,3].loglog(N_Data[1:end-1],PARA_Out_Of_Distribution_Data[4:5:40,5,ii][1:end-1],color=colors[3], linestyle="solid", marker = markers[3], fillstyle="full",label="PARA-Net (OOD)")
    ax[i,4].loglog(N_Data,FNO_Data[4:5:40,5,ii],color=colors[4], linestyle=(0,(1,1)), marker = markers[4],markersize=4, fillstyle="none",label="FNO")
    ax[i,4].loglog(N_Data,FNO_Out_Of_Distribution_Data[4:5:40,5,ii],color=colors[4], linestyle="solid", marker = markers[4],markersize=4, fillstyle="full",label="FNO (OOD)")

    # gray spines and ticks 
    for j = 1:4
        
        ax[i,j].spines["top"].set_visible(false)
        ax[i,j].spines["right"].set_visible(false)
        ax[i,j].spines["left"].set_color(tk)
        ax[i,j].spines["left"].set_linewidth(0.3)
        ax[i,j].spines["bottom"].set_color(tk)
        ax[i,j].spines["bottom"].set_linewidth(0.3)
        ax[i,j].set_xticks(N_Data[2:2:end])
        ax[i,j].set_xticklabels(["312","1250","5k","20k"])
        ax[i,j][:xaxis][:set_tick_params](colors=tk,width=0.3)
        ax[i,j][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
        ax[i,j][:yaxis][:set_tick_params](colors=tk,width=0.3)
        ax[i,j][:yaxis][:set_tick_params](which="minor",left=false)

        if i == 2
            ax[i,j].set_xlabel(latexstring("Training data ",L"N"),labelpad=2,color=lbl)
        end
    end

end

ax[1,1].set_title("PCA-Net",pad=13,color=lbl)
ax[1,2].set_title("DeepONet",pad=13,color=lbl)
ax[1,3].set_title("PARA-Net",pad=13,color=lbl)
ax[1,4].set_title("FNO",pad=13,color=lbl)

ax[1,1].set_ylabel("Navier-Stokes",color=lbl)
ax[2,1].set_ylabel("Structural mechanics",color=lbl)
ax[2,1].set_yticks([0.05, 0.1, 0.2, 0.4])
ax[2,1][:yaxis][:set_minor_formatter](plt.NullFormatter())
ax[2,1].set_yticklabels(["0.05", "0.1", "0.2", "0.4"])


plt.subplots_adjust(bottom=0.15,top=0.89,left=0.08,right=0.98,hspace=0.2)

handles, labels = ax[1,1].get_legend_handles_labels()
fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.97),ncol=4,frameon=false,fontsize=9,labelcolor="linecolor")

plt.savefig("ood"*coloroption*".pdf")
