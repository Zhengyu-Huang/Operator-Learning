using NPZ
using LinearAlgebra
using PyPlot
include("../../plotdefaults.jl")


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
    inputs   = npzread(prefix * "Random_NS_curl_f_100.npy")   
    outputs  = npzread(prefix * "Random_NS_omega_100.npy")
    
    ######################################################
    N_x, N_data = size(outputs)

    L = 1
    xx = LinRange(0, L, N_x)

    for ind in inds
        input_plot(inputs[:, :,ind],   "NS-map-input-$(ind).pdf")
        output_plot(outputs[:, :,ind], "NS-map-output-$(ind).pdf")
    end
end


####################################################
# Plot example input and output
####################################################
prefix = "/Users/elizqian/Box/HelmholtzData/data/"
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
cb1.ax.yaxis.set_tick_params(colors="#808080",width=0.3)
im2 = ax[2].pcolormesh(X, Y, outputs[:,:,sample_ind],  shading="gouraud")
cb2 = plt.colorbar(im2,ax=ax[2],shrink=0.7,aspect=15)
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
    ax[i].set_yticklabels(["0",L"\pi",L"2\pi"])
    ax[i].set_xticks([0,0.5,1])
    ax[i].set_xticklabels(["0",L"\pi",L"2\pi"])
end
ax[1].set_ylabel(L"y")
ax[1].set_title(L"\nabla\times f")
ax[2].set_title(L"\omega(T)")
fig.subplots_adjust(left=0.1,right=0.95,bottom=0.025,top=0.975,wspace=0.3)
fig.savefig("NS-map.pdf")


####################################################
# Plot median/worst case error examples
####################################################
nn_names = ["PCA", "DeepONet", "PARA", "FNO"]
ntrain = 10000
widths = [128, 128, 128, 16]
log_err = true
for ind = 2:3 # median error
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
    
    im1 = ax[1,i].pcolormesh(X, Y, inputs[:, :, ind],    shading="gouraud",vmin = imin,vmax = imax)
    im2 = ax[2,i].pcolormesh(X, Y, outputs[:, :, ind],   shading="gouraud", vmin=omin, vmax =omax)
    im3 = ax[3,i].pcolormesh(X, Y, outputs[:, :, ind+3], shading="gouraud", vmin=omin, vmax =omax)
    if log_err
        im4 = ax[4,i].pcolormesh(X,Y,broadcast(log10,err),shading="gouraud",cmap="magma",vmin=lemin,vmax=lemax)
    else
        im4 = ax[4,i].pcolormesh(X,Y,err,shading="gouraud",cmap="magma",vmin=emin,vmax=emax)
    end
    ax[1,i].set_title(nns[i],pad = 5)

    for j = 1:4
        ax[j,i].spines["top"].set_visible(false)
        ax[j,i].spines["right"].set_visible(false)
        ax[j,i].spines["left"].set_visible(false)
        ax[j,i].spines["bottom"].set_visible(false)
        ax[j,i].set_aspect("equal", "box")
        ax[j,i].set_yticks([])
        ax[j,i].set_xticks([])
    end
    
    if i == 4
        cax = fig.add_axes([0.92, 0.735368, 0.015, 0.21415],frameon=false)
        cb = plt.colorbar(im1,cax=cax, ticks=[-0.02,-0.01, 0, 0.01,0.02],drawedges=false)
        cb.outline.set_visible(false)
        cb.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

        cax2 = fig.add_axes([0.92,0.262112, 0.015, 0.450775])
        cb2 = plt.colorbar(im2,cax=cax2,ticks=[-0.2,-0.1,0,0.1,0.2])
        cb2.outline.set_visible(false)
        cb2.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

        # bottom err colorbar
        cax3 = fig.add_axes([0.92,0.02548,0.015,0.21415])
        if log_err
        cb3 = plt.colorbar(im4,cax=cax3,ticks=[-9,-7,-5,-3,-1])
        cb3.ax.set_yticklabels([L"10^{-9}",L"10^{-7}",L"10^{-5}",L"10^{-3}",L"10^{-1}"])
        else
            cb3 = plt.colorbar(im4,cax=cax3,ticks=[0, 0.025, 0.05])
        end
        cb3.outline.set_visible(false)
        cb3.ax.yaxis.set_tick_params(colors="#808080",width=0.3)
    end
end

ax[1,1].set_ylabel(L"\nabla\times f",labelpad=5)
ax[2,1].set_ylabel("True "*L"\omega(T)",labelpad=5)
ax[3,1].set_ylabel("Predicted "*L"\omega(T)",labelpad=5)
ax[4,1].set_ylabel(L"\omega(T)"*" error",labelpad=5)
plt.subplots_adjust(left = 0.05, right = 0.9, bottom = 0.025,top=.95,hspace=0.1,wspace=0.1)
if log_err
    if ind==2
        plt.savefig("NS-medians-log.pdf")
    elseif ind==3
        plt.savefig("NS-worst-log.pdf")
    end
else
    if ind==2
        plt.savefig("NS-medians.pdf")
    elseif ind==3
        plt.savefig("NS-worst.pdf")
    end
end
end

# #####################################
# ind = 3 # largest error

# # loop through all NNs once to get shared color axes
# clims = zeros((3,2))
# for i = 1:4
#     nn_name = nn_names[i]
#     inputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
#     outputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
#     inputs   = npzread(inputfile)   
#     outputs  = npzread(outputfile)

#     clims[1,1] = minimum([clims[1,1],minimum(inputs[:,:,ind])])
#     clims[1,2] = maximum([clims[1,2],maximum(inputs[:,:,ind])])
#     clims[2,1] = minimum([clims[2,1],minimum(outputs[:,:,ind])])
#     clims[2,2] = maximum([clims[2,2],maximum(outputs[:,:,ind])])
#     clims[3,1] = minimum([clims[3,1],minimum(outputs[:,:,ind+3])])
#     clims[3,2] = maximum([clims[3,2],maximum(outputs[:,:,ind+3])])
# end
# @show clims


# fig, ax = PyPlot.subplots(3,4, sharex=true, sharey=true, figsize=(6.5,4))
# for i = 1:4
#     nn_name = nn_names[i]
#     inputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
#     outputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
#     inputs   = npzread(inputfile)   
#     outputs  = npzread(outputfile)
    
#     N_x, _ = size(inputs)
#     L = 1
#     xx = LinRange(0, L, N_x)
#     Y, X = meshgrid(xx, xx)

#     vmin, vmax = minimum(outputs[:, :, ind]), maximum(outputs[:, :, ind])

#     im1 = ax[1,i].pcolormesh(X, Y, inputs[:, :, ind],    shading="gouraud",vmin = clims[1,1],vmax = clims[1,2])
#     im2 = ax[2,i].pcolormesh(X, Y, outputs[:, :, ind],   shading="gouraud", vmin=minimum([clims[2,1],clims[3,1]]), vmax =maximum([clims[2,2],clims[3,2]]))
#     im3 = ax[3,i].pcolormesh(X, Y, outputs[:, :, ind+3], shading="gouraud", vmin=minimum([clims[2,1],clims[3,1]]), vmax =maximum([clims[2,2],clims[3,2]]))

#     ax[1,i].set_title(nns[i],pad = 5)
#     # ax[3,i].set_xlabel(L"x",labelpad=10)

#     for j = 1:3
#         ax[j,i].spines["top"].set_visible(false)
#         ax[j,i].spines["right"].set_visible(false)
#         ax[j,i].spines["left"].set_visible(false)
#         ax[j,i].spines["bottom"].set_visible(false)
#         ax[j,i].set_aspect("equal", "box")
#         ax[j,i].set_yticks([])
#         ax[j,i].set_xticks([])
#     end
    
#     if i == 4
#         cax = fig.add_axes([0.92, 0.628, 0.015, 0.27],frameon=false)
#         cb = plt.colorbar(im1,cax=cax, ticks=[-0.02, 0, 0.02],drawedges=false)
#         cb.outline.set_visible(false)
#         cb.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

#         cax2 = fig.add_axes([0.92,0.025, 0.015, 0.575])
#         cb2 = plt.colorbar(im2,cax=cax2,ticks=[-0.1,0,0.1])
#         cb2.outline.set_visible(false)
#         cb2.ax.yaxis.set_tick_params(colors="#808080",width=0.3)
#     end
# end

# ax[1,1].set_ylabel(L"\nabla\times f",labelpad=5)
# ax[2,1].set_ylabel("True "*L"\omega(T)",labelpad=5)
# ax[3,1].set_ylabel("Predicted "*L"\omega(T)",labelpad=5)
# plt.subplots_adjust(left = 0.05, right = 0.9, bottom = 0.025,top=.9,hspace=0.1,wspace=0.1)
# plt.savefig("NS-worst.pdf")