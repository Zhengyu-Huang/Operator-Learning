include("Data-NN-Plot.jl") # load data

probs = ["Navier-Stokes", "Helmholtz", "Structural mechanics", "Advection"]

ylims = [1e-3 10; 1e-2 10; 0.04 0.4; 0.1 10]

fig, ax = PyPlot.subplots(ncols = 4,nrows=4, sharex=true, figsize=(6.5,5.5))

N_Data = [156, 312, 625, 1250, 2500, 5000, 10000, 20000]

for i = 1:4 # test problems
    for j = 1:4 # plot columns
        ax[i,j].loglog(N_Data, ylims[i,2]*sqrt(N_Data[1]) ./ sqrt.(N_Data), color = "#bababa",  linewidth=0.5,linestyle="dashed")
        if i == 3
            ax[i,j].text(1250,ylims[i,2]*sqrt(N_Data[1])./sqrt(1250),"1/√N",color="#bababa",fontsize=8)
        else
            ax[i,j].text(5000,ylims[i,2]*sqrt(N_Data[1])./sqrt(5000),"1/√N",color="#bababa",fontsize=8)
        end

        ax[i,j].loglog(N_Data,PCA_Data[j:5:40,5,i],color=colors[1], linestyle=(0,(1,1)), marker = markers[1], fillstyle="none",label=nns[1])
        ax[i,j].loglog(N_Data,DeepONet_Data[j:5:40,5,i],color=colors[2], linestyle=(0,(1,1)), marker = markers[2], fillstyle="none",label=nns[2])
        ax[i,j].loglog(N_Data[1:end-1],PARA_Data[j:5:35,5,i],color=colors[3], linestyle=(0,(1,1)), marker = markers[3], fillstyle="none",clip_on=false,label=nns[3])
        ax[i,j].loglog(N_Data,FNO_Data[j:5:40,5,i],color=colors[4], linestyle=(0,(1,1)), marker = markers[4], fillstyle="none",label=nns[4])

        ax[i,j].set_ylim(ylims[i,:])

        # set titles on top row
        if i == 1
            ax[i,j].set_title(sizes[j],pad=20)
        end

        # set xlabels on bottom row
        if i == 4
            ax[i,j].set_xlabel(latexstring("Training data ",L"N"),labelpad=2)
        end

        # gray spines and ticks 
        ax[i,j].spines["top"].set_visible(false)
        ax[i,j].spines["right"].set_visible(false)
        ax[i,j].spines["left"].set_color("#808080")
        ax[i,j].spines["left"].set_linewidth(0.3)
        ax[i,j].spines["bottom"].set_color("#808080")
        ax[i,j].spines["bottom"].set_linewidth(0.3)
        ax[i,j].set_xticks(N_Data[2:2:end])
        ax[i,j].set_xticklabels(["312","1250","5k","20k"])
        ax[i,j][:xaxis][:set_tick_params](colors="#808080",width=0.3)
        ax[i,j][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
        ax[i,j][:yaxis][:set_tick_params](colors="#808080",width=0.3)
        ax[i,j][:yaxis][:set_tick_params](which="minor",left=false)
        
        if i == 3
            ax[i,j].set_yticks([0.05, 0.1, 0.2, 0.4])
            ax[i,j][:yaxis][:set_minor_formatter](plt.NullFormatter())
        end

        # set labels on left column and remove tick labels elsewhere
        if j == 1
            ax[i,j].set_ylabel(probs[i])
            if i == 3
                ax[i,j].set_yticklabels(["0.05", "0.1", "0.2", "0.4"])
            end
        else
            ax[i,j].set_yticklabels("")
        end
    end
end

handles, labels = ax[4,4].get_legend_handles_labels()
plt.subplots_adjust(bottom=0.07,top=0.9,left=0.13,right=0.98,hspace=0.2)

fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.95),ncol=4,frameon=false,fontsize=9)
fig.text(0.01, 0.5, "Test Error", ha="left", va="center", rotation="vertical",fontsize=14)
plt.savefig("All-Data-Error.pdf")
plt.close("all")