# script to plot err vs. cost for all training data volumes
# can choose color option for paper plots or for darkslides, or make your own color option in plotdefaults.jl

include("Data-NN-Plot.jl") # load data

coloroption = "paper"
# coloroption = "darkslides"

include("plotdefaults.jl")


########### Error vs Cost
fig, ax = PyPlot.subplots(ncols = 4,nrows=4, sharex="col", sharey= "row", figsize=(6.8,5.5))
N_Data = [2500, 5000, 10000, 20000]
ypad = [1, 1, 2,7]

for i = 1:4 # test problems
    for j = 1:4 # plot columns

        ax[i,j].loglog(PCA_Data[(j+3)*5+1:(j+3)*5+5, 3, i], PCA_Data[(j+3)*5+1:(j+3)*5+5, 5, i], color = colors[1], linestyle=linestyle[i], marker = markers[1], fillstyle="none",      label =  nns[1]  )
        ax[i,j].loglog(DeepONet_Data[(j+3)*5+1:(j+3)*5+5, 3, i], DeepONet_Data[(j+3)*5+1:(j+3)*5+5, 5, i], color = colors[2], linestyle=linestyle[i], marker = markers[2], fillstyle="none", label =  nns[2]  )
        if j < 4 || i == 3 || i == 4
            ax[i,j].loglog(PARA_Data[(j+3)*5+1:(j+3)*5+5, 3, i], PARA_Data[(j+3)*5+1:(j+3)*5+5, 5, i], color = colors[3], linestyle=linestyle[i], marker = markers[3], fillstyle="none",     label =  nns[3]  )
        end
        ax[i,j].loglog(FNO_Data[(j+3)*5+1:(j+3)*5+5, 3, i], FNO_Data[(j+3)*5+1:(j+3)*5+5, 5, i], color = colors[4], linestyle=linestyle[i], marker = markers[4], fillstyle="none",      label =  nns[4]  )
        
        
        @info probs[i] , j, "-th column: ", " solpe : ",    [log.(FNO_Data[(j+3)*5+1:(j+3)*5+5, 3, i])   ones(5)] \ log.(FNO_Data[(j+3)*5+1:(j+3)*5+5, 5, i])
        
        # set titles on top row
        if i == 1
            ax[i,j].set_title(L"N = "*string(Int(N_Data[j])),pad=20,color=lbl)
        end

        # set xlabels on bottom row
        if i == 4
            ax[i,j].set_xlabel("Evaluation cost",labelpad=2,color=lbl)
        end

        # # gray spines and ticks 
        ax[i,j].spines["top"].set_visible(false)
        ax[i,j].spines["right"].set_visible(false)
        ax[i,j].spines["left"].set_color(tk)
        ax[i,j].spines["left"].set_linewidth(0.3)
        ax[i,j].spines["bottom"].set_color(tk)
        ax[i,j].spines["bottom"].set_linewidth(0.3)
        ax[i,j][:xaxis][:set_tick_params](colors=tk,width=0.3)
        ax[i,j][:xaxis][:set_tick_params](which="minor",bottom=false) # remove minor tick labels
        ax[i,j][:yaxis][:set_tick_params](colors=tk,width=0.3)
        ax[i,j][:yaxis][:set_tick_params](which="minor",left=false)
    
                
        if i == 3
            ax[i,j].set_yticks([0.05, 0.1, 0.2, 0.4])
            ax[i,j].set_yticklabels(["0.05","0.1","0.2","0.4"])
        elseif i == 4
            ax[i,j].set_yticks([0.1, 0.2, 0.4])
            ax[i,j].set_yticklabels(["0.1","0.2","0.4"])
        end

        ax[i,j][:yaxis][:set_minor_formatter](plt.NullFormatter())
        # set labels on left column and remove tick labels elsewhere
        if j == 1
            ax[i,j].set_ylabel(probs[i],labelpad = ypad[i],color=lbl)
        end
    end
end

handles, labels = ax[1,1].get_legend_handles_labels()
plt.subplots_adjust(bottom=0.07,top=0.9,left=0.13,right=0.98,hspace=0.2)

fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.95),ncol=4,frameon=false,fontsize=9,labelcolor="linecolor")
fig.text(0.01, 0.5, "Test Error", ha="left", va="center", rotation="vertical",fontsize=14,color=lbl)
plt.savefig("All-Error-Cost-"*coloroption*".pdf")
plt.close("all")