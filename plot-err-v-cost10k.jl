# script to plot err vs. cost for N = 10000 training data
# can choose color option for paper plots or for darkslides, or make your own color option in plotdefaults.jl

include("Data-NN-Plot.jl") # load data

coloroption = "paper"
coloroption = "darkslides"

include("plotdefaults.jl")


########### Error vs Cost
fig, ax = PyPlot.subplots(ncols = 4,nrows=1, figsize=(6,2))
N_Data = [2500, 5000, 10000, 20000]
ypad = [1, 1, 2,7]

for i = 1:4 # test problems
    j = 3 # 10k data
    ax[i].loglog(PCA_Data[(j+3)*5+1:(j+3)*5+5, 3, i], PCA_Data[(j+3)*5+1:(j+3)*5+5, 5, i], color = colors[1], linestyle=linestyle[3], marker = markers[1], fillstyle="none",      label =  nns[1]  )
    ax[i].loglog(DeepONet_Data[(j+3)*5+1:(j+3)*5+5, 3, i], DeepONet_Data[(j+3)*5+1:(j+3)*5+5, 5, i], color = colors[2], linestyle=linestyle[3], marker = markers[2], fillstyle="none", label =  nns[2]  )
    if j < 4 || i == 3 || i == 4
        ax[i].loglog(PARA_Data[(j+3)*5+1:(j+3)*5+5, 3, i], PARA_Data[(j+3)*5+1:(j+3)*5+5, 5, i], color = colors[3], linestyle=linestyle[3], marker = markers[3], fillstyle="none",     label =  nns[3]  )
    end
    ax[i].loglog(FNO_Data[(j+3)*5+1:(j+3)*5+5, 3, i], FNO_Data[(j+3)*5+1:(j+3)*5+5, 5, i], color = colors[4], linestyle=linestyle[3], marker = markers[4], fillstyle="none",      label =  nns[4]  )
        
        
    # set titles on top row
    ax[i].set_title(probs[i],pad=15,fontsize=10,color=lbl)

    # set xlabels on bottom row
    ax[i].set_xlabel("Evaluation cost",labelpad=2,color=lbl)

    # # gray spines and ticks 
    ax[i].spines["top"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["left"].set_color(tk)
    ax[i].spines["left"].set_linewidth(0.3)
    ax[i].spines["bottom"].set_color(tk)
    ax[i].spines["bottom"].set_linewidth(0.3)
    ax[i][:xaxis][:set_tick_params](colors=tk,width=0.3)
    ax[i][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
    ax[i][:yaxis][:set_tick_params](colors=tk,width=0.3)
    ax[i][:yaxis][:set_tick_params](which="minor",left=false)
    
                
    if i == 3
        ax[i].set_yticks([0.05, 0.1, 0.2, 0.4])
        ax[i].set_yticklabels(["0.05","0.1","0.2","0.4"])
    elseif i == 4
        ax[i].set_yticks([0.1, 0.2, 0.4])
        ax[i].set_yticklabels(["0.1","0.2","0.4"])
    end

    ax[i][:yaxis][:set_minor_formatter](plt.NullFormatter())

end
ax[1].set_ylabel("Test error",color=lbl)

handles, labels = ax[1].get_legend_handles_labels()
plt.subplots_adjust(bottom=0.2,top=0.8,left=0.09,right=0.99,wspace=0.33)

fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.93),ncol=4,frameon=false,fontsize=8,handlelength=0,labelcolor="linecolor")
# fig.text(0.01, 0.5, "Test Error", ha="left", va="center", rotation="vertical",fontsize=14)
plt.savefig("tenk-Error-Cost-"*coloroption*".pdf")
plt.close("all")