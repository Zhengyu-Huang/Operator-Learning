using NPZ
using LinearAlgebra
using PyPlot

# script to plot advection input-output map and median/maximum error cases
# can choose color option for paper plots or for darkslides, or make your own color option in plotdefaults.jl

include("Data-NN-Plot.jl") # load data

coloroption = "paper"
# coloroption = "darkslides"

include("../../plotdefaults.jl")

function meshgrid(xin, yin)
  return  xin' .* ones(length(yin)) , ones(length(xin))' .* yin
end


####################################################
# Plot example input and output
####################################################
# prefix = "/Users/elizqian/Box/HelmholtzData/data/"
prefix = "../../data/"
inputs   = npzread(prefix * "Random_NS_curl_f_100.npy")   
outputs  = npzread(prefix * "Random_NS_omega_100.npy")
sample_ind = 1
L = 1
N_x, _ = size(inputs[:,:,sample_ind])
xx = LinRange(0, L, N_x)
Y, X = meshgrid(xx, xx)

fig,ax = PyPlot.subplots(1,2,sharex=true,sharey=true,figsize=(4.5,2))
im1 = ax[1].pcolormesh(X, Y, inputs[:,:,sample_ind],  shading="gouraud")
cb1 = plt.colorbar(im1,ax=ax[1],shrink=0.7,aspect=15)
cb1.outline.set_visible(false)
cb1.ax.yaxis.set_tick_params(colors=tk,width=0.3)
im2 = ax[2].pcolormesh(X, Y, outputs[:,:,sample_ind],  shading="gouraud")
cb2 = plt.colorbar(im2,ax=ax[2],shrink=0.7,aspect=15)
cb2.outline.set_visible(false)
cb2.ax.yaxis.set_tick_params(colors=tk,width=0.3)

for i = 1:2
    ax[i].set_aspect("equal","box")
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].spines["top"].set_visible(false)
    ax[i][:xaxis][:set_tick_params](colors=tk,width=0.3)
    ax[i][:yaxis][:set_tick_params](colors=tk,width=0.3)
    ax[i].set_xlabel(L"x",color=lbl)
    ax[i].set_yticks([0,0.5,1])
    ax[i].set_yticklabels(["0",L"\pi",L"2\pi"])
    ax[i].set_xticks([0,0.5,1])
    ax[i].set_xticklabels(["0",L"\pi",L"2\pi"])
end
ax[1].set_ylabel(L"y",color=lbl)
ax[1].set_title(L"f'",color=lbl)
ax[2].set_title(L"\omega(T)",color=lbl)
fig.subplots_adjust(left=0.1,right=0.92,bottom=0.08,top=0.975,wspace=0.3)
fig.savefig("NS-map.pdf")


####################################################
# Plot median/worst case error examples
####################################################
nn_names = ["PCA", "DeepONet", "PARA", "FNO"]
ntrain = 10000
widths = [128, 128, 128, 16]
log_err = true
for ind = 2:3 # median error
    ims = Array{Any}(undef,4,4)
fig, ax = PyPlot.subplots(4,4, sharex=true, sharey=true, figsize=(6.5,6))
for i = 1:4
    nn_name = nn_names[i]
    inputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
    outputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    
    N_x, _ = size(inputs)
    L = 1
    xx = LinRange(0, L, N_x)
    Y, X = meshgrid(xx, xx)

    imin = -0.027
    imax = 0.027
    omin = -0.2
    omax =  0.2
    lemin = -9
    lemax = -1
    emin = 0
    emax = 0.06

    @show minimum(inputs[:,:,ind]), maximum(inputs[:,:,ind])
    @show minimum(minimum([outputs[:,:,ind] outputs[:,:,ind+3]])),maximum(maximum([outputs[:,:,ind] outputs[:,:,ind+3]]))
    err = broadcast(abs,outputs[:,:,ind]-outputs[:,:,ind+3])
    @show minimum(err), maximum(err)
    
    ims[1,i] = ax[1,i].pcolormesh(X, Y, inputs[:, :, ind],    shading="gouraud",vmin = imin,vmax = imax)
    ims[2,i] = ax[2,i].pcolormesh(X, Y, outputs[:, :, ind],   shading="gouraud", vmin=omin, vmax =omax)
    ims[3,i] = ax[3,i].pcolormesh(X, Y, outputs[:, :, ind+3], shading="gouraud", vmin=omin, vmax =omax)
    if log_err
        ims[4,i] = ax[4,i].pcolormesh(X,Y,broadcast(log10,err),shading="gouraud",cmap="magma",vmin=lemin,vmax=lemax)
    else
        ims[4,i] = ax[4,i].pcolormesh(X,Y,err,shading="gouraud",cmap="magma",vmin=emin,vmax=emax)
    end
    ax[1,i].set_title(nns[i],pad = 5,fontsize=16,color=lbl)

    for j = 1:4
        ax[j,i].spines["top"].set_visible(false)
        ax[j,i].spines["right"].set_visible(false)
        ax[j,i].spines["left"].set_visible(false)
        ax[j,i].spines["bottom"].set_visible(false)
        ax[j,i].set_aspect("equal", "box")
        ax[j,i].set_yticks([])
        ax[j,i].set_xticks([])
    end
end

ax[1,1].set_ylabel(L"f'",labelpad=5,fontsize=14,color=lbl)
ax[2,1].set_ylabel("True "*L"\omega(T)",labelpad=5,fontsize=14,color=lbl)
ax[3,1].set_ylabel("Predicted "*L"\omega(T)",labelpad=5,fontsize=14,color=lbl)
ax[4,1].set_ylabel(L"\omega(T)"*" error",labelpad=5,fontsize=14,color=lbl)
plt.subplots_adjust(left = 0.05, right = 0.86, bottom = 0.025,top=.95,hspace=0.1,wspace=0.1)

temp = ax[1,4].get_position()
xw = temp.x1-temp.x0
cax = fig.add_axes([temp.x1+0.1*xw, temp.y0, 0.1*xw, temp.y1-temp.y0],frameon=false)
cb = plt.colorbar(ims[1,4],cax=cax, ticks=[-0.02, 0, 0.02],drawedges=false)
cb.outline.set_visible(false)
cb.ax.yaxis.set_tick_params(colors=tk,width=0.3)

temp = ax[2,4].get_position()
temp2 = ax[3,4].get_position()
xw = temp.x1-temp.x0
cax2 = fig.add_axes([temp.x1+0.1*xw,temp2.y0, 0.1*xw, temp.y1-temp2.y0])
cb2 = plt.colorbar(ims[2,4],cax=cax2,ticks=[-0.2,-0.1,0,0.1,0.2])
cb2.outline.set_visible(false)
cb2.ax.yaxis.set_tick_params(colors=tk,width=0.3)

temp = ax[4,4].get_position()
cax3 = fig.add_axes([temp.x1+0.1*xw,temp.y0,0.1*xw, temp.y1-temp.y0])
if log_err
    cb3 = plt.colorbar(ims[4,4],cax = cax3,ticks=[-9,-7,-5,-3,-1])
    cb3.ax.set_yticklabels([L"10^{-9}",L"10^{-7}",L"10^{-5}",L"10^{-3}",L"10^{-1}"])
else
    cb3 = plt.colorbar(im4,cax=cax3,ticks=[0, 0.025, 0.05])
end
cb3.outline.set_visible(false)
cb3.ax.yaxis.set_tick_params(colors=tk,width=0.3)

if log_err
    if ind==2
        plt.savefig("NS-medians-log-"*coloroption*".jpg",dpi=300)
    elseif ind==3
        plt.savefig("NS-worst-log-"*coloroption*".jpg",dpi=300)
    end
else
    if ind==2
        plt.savefig("NS-medians.pdf")
    elseif ind==3
        plt.savefig("NS-worst.pdf")
    end
end
end

plt.close("all")