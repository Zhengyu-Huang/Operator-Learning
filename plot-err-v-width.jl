
########### Error vs Width
# ylims = [1e-3 10; 1e-2 10; 0.04 0.4; 0.1 10]

fig, ax = PyPlot.subplots(ncols = 4,nrows=4, sharex="col", sharey= "row", figsize=(6.5,5.5))
N_Data = [2500, 5000, 10000, 20000]
N_Width = [16, 64, 128, 256, 512]
N_Df = [2, 4, 8, 16, 32] 
ypad = [1, 1, 2,7]
for i = 1:4 # test problems
    for j = 1:4 # loop N 

        
        ax[i,1].loglog(N_Width,PCA_Data[(j+3)*5+1:(j+3)*5+5,5,i],color=colors[1], linestyle=linestyle[j], marker = markers[j], fillstyle="none",label =  L"N = "*string(Int(N_Data[j])))
        ax[i,2].loglog(N_Width,DeepONet_Data[(j+3)*5+1:(j+3)*5+5,5,i],color=colors[2], linestyle=linestyle[j], marker = markers[j], fillstyle="none",label =  L"N = "*string(Int(N_Data[j])))
        if j < 4 || i == 3 || i == 4
            ax[i,3].loglog(N_Width,PARA_Data[(j+3)*5+1:(j+3)*5+5,5,i],color=colors[3], linestyle=linestyle[j], marker = markers[j], fillstyle="none",label=L"N = "*string(Int(N_Data[j])))
        end
        ax[i,4].loglog(N_Df,FNO_Data[(j+3)*5+1:(j+3)*5+5,5,i],color=colors[4], linestyle=linestyle[j], marker = markers[j], fillstyle="none",label=L"N = "*string(Int(N_Data[j])))

        # ax[i,j].set_ylim(ylims[i,:])

        # set titles on top row
        if i == 1
            ax[i,j].set_title(nns[j],pad=20)
        end

        # set xlabels on bottom row
        if i == 4
            if j == 4
                ax[i,j].set_xlabel(latexstring("Lifting dimension "*L"d_f"),labelpad=2)
            else
                ax[i,j].set_xlabel(latexstring("Network width ",L"w"),labelpad=2)
            end
        end

        # gray spines and ticks 
        ax[i,j].spines["top"].set_visible(false)
        ax[i,j].spines["right"].set_visible(false)
        ax[i,j].spines["left"].set_color("#808080")
        ax[i,j].spines["left"].set_linewidth(0.3)
        ax[i,j].spines["bottom"].set_color("#808080")
        ax[i,j].spines["bottom"].set_linewidth(0.3)
        # ax[i,j].set_xticklabels(["16","","128","256","512"])
        ax[i,j][:xaxis][:set_tick_params](colors="#808080",width=0.3)
        ax[i,j][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
        ax[i,j][:yaxis][:set_tick_params](colors="#808080",width=0.3)
        ax[i,j][:yaxis][:set_tick_params](which="minor",left=false)
        ax[i,j][:yaxis][:set_minor_formatter](plt.NullFormatter())
        
        if i == 3
            ax[i,j].set_yticks([0.05, 0.1, 0.2, 0.4])
            ax[i,j].set_yticklabels(["0.05","0.1","0.2","0.4"])
        elseif i == 4
            ax[i,j].set_yticks([0.1, 0.2, 0.4])
            ax[i,j].set_yticklabels(["0.1","0.2","0.4"])
        end

        # set labels on left column and remove tick labels elsewhere
        if j == 1
            ax[i,j].set_ylabel(probs[i],labelpad = ypad[i])
        end
    end
end

for i = 1:4
    for j = 1:4
        if j == 4
            ax[i, j].set_xticks(FNO_Data[(j+3)*5+1:(j+3)*5+5, 2, 1])
            ax[i, j].set_xticklabels(["2","4","8","16","32"])
        else
            ax[i, j].set_xticks(N_Width)
	    ax[i, j].set_xticklabels=(["16",L"64\,","128","","512"])
        end

    end
end

handles, labels = ax[1,1].get_legend_handles_labels()
plt.subplots_adjust(bottom=0.07,top=0.9,left=0.13,right=0.98,hspace=0.2)

fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.95),ncol=4,frameon=false,fontsize=9)
fig.text(0.01, 0.5, "Test Error", ha="left", va="center", rotation="vertical",fontsize=14)
plt.savefig("All-Error-Width.pdf")
plt.close("all")


