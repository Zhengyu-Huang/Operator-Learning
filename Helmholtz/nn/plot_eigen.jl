using NPZ
using LinearAlgebra
using PyPlot

# script to plot Helmholtz worst case eigenmode

coloroption = "paper"
# coloroption = "darkslides"

include("../../plotdefaults.jl")


function meshgrid(xin, yin)
    return  xin' .* ones(length(yin)) , ones(length(xin))' .* yin
end
  
nn_names = ["PCA", "DeepONet", "PARA", "FNO"]
ntrain = 10000
widths = [128, 128, 128, 16]
  
# prefix = "../../data/"
# inputs   = npzread(prefix * "Random_Helmholtz_high_cs_100.npy")   
# outputs  = npzread(prefix * "Random_Helmholtz_high_K_100.npy") #this is actuallythe solution u
eigen = npzread("eigen.npy")
sample_ind = 1
L = 1
N_x, _ = size(eigen)
xx = LinRange(0, L, N_x)
Y, X = meshgrid(xx, xx)


ims = Array{Any}(undef,4,4)
fig,ax = PyPlot.subplots(1,6,sharey=true,figsize=(8.5,2),gridspec_kw=Dict("width_ratios"=> [1,1,1,1,0.2,1]))
im1 = ax[6].pcolormesh(X, Y, eigen/norm(eigen),vmin=-0.036,vmax=0.036,shading="gouraud",cmap="RdBu")
ax[6].set_title("Smallest eigenmode")

ind = 3 # worst case
for i = 1:4
    nn_name = nn_names[i]
    inputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
    outputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    # err = broadcast(abs,outputs[:,:,ind]-outputs[:,:,ind+3])
    err = outputs[:,:,ind]-outputs[:,:,ind+3]
    ims[i] = ax[i].pcolormesh(X, Y, err/norm(err),vmin=-0.036,vmax=0.036,shading="gouraud",cmap="RdBu")
    ax[i].set_title(nns[i])
    @show maximum(maximum(err/norm(err))), minimum(minimum(err/norm(err)))
end

for i = 1:6
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].spines["top"].set_visible(false)

    if i!=5
        ax[i].set_aspect("equal","box")
        ax[i][:xaxis][:set_tick_params](colors=tk,width=0.3)
        ax[i][:yaxis][:set_tick_params](colors=tk,width=0.3)
        # ax[i].set_xlabel(L"x",color=lbl)
        ax[i].set_yticks([0,0.5,1])
        ax[i].set_xticks([0,0.5,1])
    else
        # ax[i].set_yticks([])
        # ax[i].set_xticks([])
        ax[i][:xaxis][:set_visible](false)
        ax[i][:yaxis][:set_visible](false)
    end
end

fig.subplots_adjust(left=0.05,right=0.91,bottom=0.01,top=0.98,wspace=0.1)
temp = ax[6].get_position()
xw = temp.x1-temp.x0
cax = fig.add_axes([temp.x1+0.1*xw,temp.y0, 0.1*xw, temp.y1-temp.y0])
cb1 = plt.colorbar(im1,cax=cax,ticks=[-0.02,0,0.02])
cb1.outline.set_visible(false)
cb1.ax.yaxis.set_tick_params(colors=tk,width=0.3)

fig.savefig("Helmholtz-worst-eigen-"*coloroption*".pdf")
plt.close("all")
