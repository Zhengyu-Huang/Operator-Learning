using NPZ
using LinearAlgebra
using PyPlot
include("../../plotdefaults.jl")

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 10
    font0 = Dict(
    "font.size" => 12,          # title
    "axes.labelsize" => 12, # axes labels
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
merge!(rcParams, font0)

function meshgrid(xin, yin)
  return  xin' .* ones(length(yin)) , ones(length(xin))' .* yin
end


function input_plot(data, file_name)
    
    L = 1
    N_x, _ = size(data)
    xx = LinRange(0, L, N_x)
    Y, X = meshgrid(xx, xx)
    
    fig = figure()
    
    pcolormesh(X, Y, data,  shading="gouraud")

    colorbar()
    fig.tight_layout()
    fig.savefig(file_name)
end


function output_plot(data, file_name)
    
    L = 1
    N_x, _ = size(data)
    xx = LinRange(0, L, N_x)
    Y, X = meshgrid(xx, xx)
    
    fig = figure()
    
    pcolormesh(X, Y, data,  shading="gouraud")
    
    colorbar()
    fig.tight_layout()
    fig.savefig(file_name)
end

function map_plot(prefix = "../../data/", inds = [1,11])
    inputs   = npzread(prefix * "Random_Helmholtz_high_cs_100.npy")   
    outputs  = npzread(prefix * "Random_Helmholtz_high_K_100.npy")
    
    ######################################################
    N_x, N_data = size(outputs)

    L = 1
    xx = LinRange(0, L, N_x)

    for ind in inds
        input_plot(inputs[:, :, ind],   "Helmholtz-map-input-$(ind).pdf")
        output_plot(outputs[:, :, ind], "Helmholtz-map-output-$(ind).pdf")
    end
end




# i = 0, 1, 2 smallest, median, largest
function prediction_plot(nn_name, ntrain, width, ind)
    err = ["s", "m", "l"]
    inputfile  = nn_name * "/" * string(ntrain) * "_" * string(width) * "_test_input_save.npy"
    outputfile = nn_name * "/" * string(ntrain) * "_" * string(width) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    
    N_x, _ = size(inputs)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig, ax = PyPlot.subplots(ncols = 3, sharex=true, sharey=true, figsize=(18,6))
    
    Y, X = meshgrid(xx, xx)
    
    vmin, vmax = minimum(outputs[:, :, ind]), maximum(outputs[:, :, ind])
    ax[1].pcolormesh(X, Y, inputs[:, :, ind],    shading="gouraud")
    ax[2].pcolormesh(X, Y, outputs[:, :, ind],   shading="gouraud", vmin=vmin, vmax =vmax)
    ax[3].pcolormesh(X, Y, outputs[:, :, ind+3], shading="gouraud", vmin=vmin, vmax =vmax)

    
    fig.tight_layout()
    fig.savefig("Helmholtz-" * err[ind] * "_" * nn_name * "_" * string(ntrain) * "_" * string(width)* ".pdf")
    
end

prefix = "../../data/"
inputs   = npzread(prefix * "Random_Helmholtz_high_cs_100.npy")   
outputs  = npzread(prefix * "Random_Helmholtz_high_K_100.npy") #this is actuallythe solution u
sample_ind = 1
L = 1
N_x, _ = size(inputs[:,:,sample_ind])
xx = LinRange(0, L, N_x)
Y, X = meshgrid(xx, xx)

fig,ax = PyPlot.subplots(1,2,sharex=true,sharey=true,figsize=(4.5,2))
im1 = ax[1].pcolormesh(X, Y, inputs[:,:,sample_ind],  shading="gouraud")
cb1 = plt.colorbar(im1,ax=ax[1],shrink=0.68,aspect=15)
cb1.outline.set_visible(false)
cb1.ax.yaxis.set_tick_params(colors="#808080",width=0.3)
im2 = ax[2].pcolormesh(X, Y, outputs[:,:,sample_ind],  shading="gouraud")
cb2 = plt.colorbar(im2,ax=ax[2],shrink=0.68,aspect=15,ticks=[-0.03, 0, 0.03])
cb2.outline.set_visible(false)
cb2.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

for i = 1:2
    ax[i].set_aspect("equal","box")
    ax[i].spines["left"].set_visible(false)
    ax[i].spines["right"].set_visible(false)
    ax[i].spines["bottom"].set_visible(false)
    ax[i].spines["top"].set_visible(false)
    ax[i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[i].set_xlabel(L"x")
    ax[i].set_yticks([0,0.5,1])
    # ax[i].set_yticklabels(["0",L"\pi",L"2\pi"])
    ax[i].set_xticks([0,0.5,1])
    # ax[i].set_xticklabels(["0",L"\pi",L"2\pi"])
end
ax[1].set_ylabel(L"y")
ax[1].set_title(L"c")
ax[2].set_title(L"u")
fig.subplots_adjust(left=0.13,right=0.92,bottom=0.07,top=0.98,wspace=0.3)
fig.savefig("Helmholtz-map.pdf")

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 12
    font0 = Dict(
    "font.size" => 12,          # title
    "axes.labelsize" => 12, # axes labels
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
merge!(rcParams, font0)


nn_names = ["PCA", "DeepONet", "PARA", "FNO"]
ntrain = 10000
widths = [128, 128, 128, 16]

imin = 19.5
imax = 20.4
omin = -0.1
omax = 0.1

log_err = true

for ind = 2:3 # median error

ims = Array{Any}(undef,4,4)
fig, ax = PyPlot.subplots(4,4, sharex=true, sharey=true, figsize=(6.5,6.5))
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

    vmin, vmax = minimum(outputs[:, :, ind]), maximum(outputs[:, :, ind])
    @show vmin,vmax
    err = broadcast(abs,outputs[:,:,ind]-outputs[:,:,ind+3])
    @show minimum(err), maximum(err)

    ims[1,i] = ax[1,i].pcolormesh(X, Y, inputs[:, :, ind],    shading="gouraud",vmin=imin,vmax=imax)
    ims[2,i] = ax[2,i].pcolormesh(X, Y, outputs[:, :, ind],   shading="gouraud", vmin=omin, vmax =omax)
    ims[3,i] = ax[3,i].pcolormesh(X, Y, outputs[:, :, ind+3], shading="gouraud", vmin=omin, vmax =omax)
    if log_err
        if ind == 2
            ims[4,i] = ax[4,i].pcolormesh(X, Y, broadcast(log10,err),shading="gouraud",vmin=-10,vmax=-2,cmap="magma")
        elseif ind == 3
            ims[4,i] = ax[4,i].pcolormesh(X, Y, broadcast(log10,err),shading="gouraud",vmin=-6,vmax=-1,cmap="magma")
        end
    else
        if ind == 2
            ims[4,i] = ax[4,i].pcolormesh(X, Y, err,shading="gouraud",vmin=0,vmax=0.003,cmap="magma")
        elseif ind == 3
            ims[4,i] = ax[4,i].pcolormesh(X, Y, err,shading="gouraud",vmin=0,vmax=0.1,cmap="magma")
        end
    end
    ax[1,i].set_title(nns[i],pad = 5,fontsize=16)

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

ax[1,1].set_ylabel(L"c",labelpad=5,fontsize=14)
ax[2,1].set_ylabel("True "*L"u",labelpad=5,fontsize=14)
ax[3,1].set_ylabel("Predicted "*L"u",labelpad=5,fontsize=14)
ax[4,1].set_ylabel("Error in "*L"u",labelpad=5,fontsize=14)

plt.subplots_adjust(left = 0.05, right = 0.87, bottom = 0.025,top=0.95,hspace=0.1,wspace=0.1)

temp = ax[1,4].get_position()
xw = temp.x1-temp.x0
cax1 = fig.add_axes([temp.x1+0.1*xw,temp.y0,0.1*xw,temp.y1-temp.y0])
cb1 = plt.colorbar(ims[1,4],cax=cax1)
cb1.outline.set_visible(false)
cb1.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

temp = ax[2,4].get_position()
temp2 = ax[3,4].get_position()
xw = temp.x1-temp.x0
cax2 = fig.add_axes([temp.x1+0.1*xw,temp2.y0, 0.1*xw, temp.y1-temp2.y0])
cb2 = plt.colorbar(ims[2,4],cax=cax2,ticks=[-0.1,-0.05,0,0.05,0.1])
cb2.outline.set_visible(false)
cb2.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

temp = ax[4,4].get_position()
cax3 = fig.add_axes([temp.x1+0.1*xw,temp.y0,0.1*xw, temp.y1-temp.y0])
if log_err
    cb3 = plt.colorbar(ims[4,4],cax = cax3,ticks = [-10,-8,-6,-4,-2,0,2])
    cb3.ax.set_yticklabels([L"10^{-10}",L"10^{-8}",L"10^{-6}",L"10^{-4}",L"10^{-2}",L"10^0",L"10^2"])
else
    if ind == 2
        cb3 = plt.colorbar(ims[4,4],cax = cax3,ticks = [0, 0.001,0.002,0.003])
    elseif ind == 3
        cb3 = plt.colorbar(ims[4,4],cax = cax3,ticks = [0, 0.05, 0.1])
    end
end
cb3.outline.set_visible(false)
cb3.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

if ind == 2
if log_err
    plt.savefig("Helmholtz-medians-log.pdf")
else
    plt.savefig("Helmholtz-medians.pdf")
end
elseif ind == 3
    if log_err
        plt.savefig("Helmholtz-worst-log.pdf")
    else
        plt.savefig("Helmholtz-worst.pdf")
    end
end
end

#############################################
ind = 3 # median error

# loop through all NNs once to get shared color axes
clims = zeros((3,2))
clims[1,:] = [20,20]
for i = 1:4
    nn_name = nn_names[i]
    inputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
    outputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)

    clims[1,1] = minimum([clims[1,1],minimum(inputs[:,:,ind])])
    clims[1,2] = maximum([clims[1,2],maximum(inputs[:,:,ind])])
    clims[2,1] = minimum([clims[2,1],minimum(outputs[:,:,ind])])
    clims[2,2] = maximum([clims[2,2],maximum(outputs[:,:,ind])])
    clims[3,1] = minimum([clims[3,1],minimum(outputs[:,:,ind+3])])
    clims[3,2] = maximum([clims[3,2],maximum(outputs[:,:,ind+3])])
end
@show clims
