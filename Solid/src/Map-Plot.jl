using NPZ
using LinearAlgebra
using PyPlot
include("../../plotdefaults.jl")

using Random, Distributions
include("PlatePull.jl")



function map_plot(prefix = "../../data/", inds = [1000,1201])

    inputs   = npzread(prefix * "Random_UnitCell_Fn_100.npy")   
    outputs  = npzread(prefix * "Random_UnitCell_sigma_100.npy")
    XY = npzread(prefix * "Random_UnitCell_XY_100.npy")
    
    
    N = 21
    xgrid = LinRange(0,1,N)
    
    porder = 2
    θ = rand(Normal(0, 1.0), 100);
    filename = "square-circle-coarse-o2"
    domain, Fn = ConstructDomain(porder, θ, filename)
    ngp = Int64(sqrt(length(domain.elements[1].weights)))
    
    
    for ind in inds
        fig, ax = PyPlot.subplots()
        ax.plot(xgrid, inputs[:, ind], "--o", fillstyle="none")
        fig.tight_layout()
        fig.savefig("Solid-map-input-$(ind).pdf")

        
        fig, ax = PyPlot.subplots()
        visσ(domain, ngp, σ=outputs[:, ind], ax = ax, mycolorbar=true)
        fig.savefig("Solid-map-output-$(ind).pdf")

    end 
end

# i = 0, 1, 2 smallest, median, largest
function prediction_plot(nn_name, ntrain, width, ind)
    err = ["s", "m", "l"]
    inputfile  = "../nn/" * nn_name * "/" * string(ntrain) * "_" * string(width) * "_test_input_save.npy"
    outputfile = "../nn/" * nn_name * "/" * string(ntrain) * "_" * string(width) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    
    porder = 2
    θ = rand(Normal(0, 1.0), 100);
    filename = "square-circle-coarse-o2"
    domain, Fn = ConstructDomain(porder, θ, filename)
    ngp = Int64(sqrt(length(domain.elements[1].weights)))
    
    
    N_x, _ = size(inputs)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig, ax = PyPlot.subplots(ncols = 3, sharex=true, sharey=false, figsize=(18,6))
    ax[1].plot(xx, inputs[:, ind], "--o", fillstyle="none")
    
    vmin, vmax = minimum(outputs[:, ind]), maximum(outputs[:, ind])
    visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind],     ax = ax[2], mycolorbar=false )
    visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind + 3], ax = ax[3], mycolorbar=false )

    
    @info "error is: ", norm(outputs[:, ind+3] - outputs[:, ind])/norm(outputs[:, ind])
    
    fig.tight_layout()
    fig.savefig("Solid-" * err[ind] * "_" * nn_name * "_" * string(ntrain) * "_" * string(width)* ".pdf")
    plt.close()
    
end


# map_plot()
# # prediction_plot("PCA", 10000, 128, 1)
# prediction_plot("PCA", 10000, 128, 2)
# prediction_plot("PCA", 10000, 128, 3)

# # prediction_plot("FNO", 10000, 16, 1)
# prediction_plot("FNO", 10000, 16, 2)
# prediction_plot("FNO", 10000, 16, 3)


# # prediction_plot("DeepONet", 10000, 128, 1)
# prediction_plot("DeepONet", 10000, 128, 2)
# prediction_plot("DeepONet", 10000, 128, 3)


# # prediction_plot("PARA", 10000, 128, 1)
# prediction_plot("PARA", 10000, 128, 2)
# prediction_plot("PARA", 10000, 128, 3)

ntrain = 10000
widths = [128, 128, 128, 16]

ind = 2 # medians

fig,ax = PyPlot.subplots(3,4,sharex=true,figsize=(6.5,4))
fig2,ax2 = PyPlot.subplots(4,4,sharex=true,figsize = (6.5,6))
for i = 1:4
    nn_name = nn_names[i]
    inputfile  = "../nn/" * nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
    outputfile = "../nn/" * nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    
    porder = 2
    θ = rand(Normal(0, 1.0), 100);
    filename = "square-circle-coarse-o2"
    domain, Fn = ConstructDomain(porder, θ, filename)
    ngp = Int64(sqrt(length(domain.elements[1].weights)))
    
    N_x, _ = size(inputs)
    L = 1
    xx = LinRange(0, L, N_x)
    ax[1,i].plot(xx, inputs[:, ind], "--o",color="#808080", fillstyle="none")
    ax[1,i].set_title(nns[i],pad = 5)
    ax[1,i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[1,i].set_xticks([])
    ax[1,i].set_ylim([-300,400])
    
    # vmin, vmax = minimum(outputs[:, ind]), maximum(outputs[:, ind])
    vmin = 0
    vmax = 350
    visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind],     ax = ax[2,i], mycolorbar=false )
    im3 = visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind + 3], ax = ax[3,i], mycolorbar=false )

    ax[1,i].spines["top"].set_visible(false)
    ax[1,i].spines["right"].set_visible(false)
    ax[1,i].spines["left"].set_color("#808080")
    ax[1,i].spines["left"].set_linewidth(0.3)
    ax[1,i].spines["bottom"].set_color("#808080")
    ax[1,i].spines["bottom"].set_linewidth(0.3)
    if i > 1
        ax[1,i].set_yticklabels([])
    end

    for j = 2:3
        ax[j,i].spines["top"].set_visible(false)
        ax[j,i].spines["right"].set_visible(false)
        ax[j,i].spines["left"].set_visible(false)
        ax[j,i].spines["bottom"].set_visible(false)
        ax[j,i].set_aspect("equal","box")
        ax[j,i].set_yticks([])
        ax[j,i].set_xticks([])
    end

    if i == 4
        cax2 = fig.add_axes([0.92,0.025, 0.015, 0.575])
        cb2 = plt.colorbar(im3,cax=cax2,ticks=[0,50,100,150,200,250,300,350])
        cb2.outline.set_visible(false)
        cb2.ax.yaxis.set_tick_params(colors="#808080",width=0.3)
    end

    ##### 
    # second median plot with error
    ###############################
    ax2[1,i].plot(xx, inputs[:, ind], "--o",color="#808080", fillstyle="none")
    ax2[1,i].set_title(nns[i],pad = 5)
    ax2[1,i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax2[1,i].set_xticks([])
    ax2[1,i].set_ylim([-300,400])
    ax2[1,i].spines["top"].set_visible(false)
    ax2[1,i].spines["right"].set_visible(false)
    ax2[1,i].spines["left"].set_color("#808080")
    ax2[1,i].spines["left"].set_linewidth(0.3)
    ax2[1,i].spines["bottom"].set_color("#808080")
    ax2[1,i].spines["bottom"].set_linewidth(0.3)
    if i > 1
        ax2[1,i].set_yticklabels([])
    end

    visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind],     ax = ax2[2,i], mycolorbar=false )
    im3 = visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind + 3], ax = ax2[3,i], mycolorbar=false )

    im4 = visσ(domain, ngp, -60, 50; σ=(outputs[:, ind + 3]-outputs[:,ind]), ax = ax2[4,i], mycolorbar=false )

    for j = 2:4
        ax2[j,i].spines["top"].set_visible(false)
        ax2[j,i].spines["right"].set_visible(false)
        ax2[j,i].spines["left"].set_visible(false)
        ax2[j,i].spines["bottom"].set_visible(false)
        ax2[j,i].set_aspect("equal","box")
        ax2[j,i].set_yticks([])
        ax2[j,i].set_xticks([])
    end


    if i == 4
        cax2 = fig2.add_axes([0.92,0.255, 0.015, 0.41])
        cb2 = plt.colorbar(im3,cax=cax2,ticks=[0,50,100,150,200,250,300,350])
        cb2.outline.set_visible(false)
        cb2.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

        cax = fig2.add_axes([0.92,0.035, 0.015, 0.185])
        cb = plt.colorbar(im4,cax=cax,ticks=[-50, -25, 0, 25, 50])
        cb.outline.set_visible(false)
        cb.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

    end
end
ax[1,1].set_ylabel(L"\tau",labelpad=5)
ax[2,1].set_ylabel("True displacement",labelpad=5)
ax[3,1].set_ylabel("Predicted displacement",labelpad=5)
fig.subplots_adjust(left = 0.05, right = 0.9, bottom = 0.025,top=.9,hspace=0.1,wspace=0.1)
fig.savefig("Solid-medians.pdf")

ax2[1,1].set_ylabel(L"\tau",labelpad=1)
ax2[2,1].set_ylabel("True displacement",labelpad=24)
ax2[3,1].set_ylabel("Predicted displacement",labelpad=24)
ax2[4,1].set_ylabel("Displacement error",labelpad=24)
fig2.subplots_adjust(left = 0.08, right = 0.9, bottom = 0.025,top=.9,hspace=0.1,wspace=0.1)
fig2.savefig("Solid-medians-err.pdf")
# plt.close()

ind = 3 # worst case
fig,ax = PyPlot.subplots(3,4,sharex=true,figsize=(6.5,4))
for i = 1:4
    nn_name = nn_names[i]
    inputfile  = "../nn/" * nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
    outputfile = "../nn/" * nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    
    porder = 2
    θ = rand(Normal(0, 1.0), 100);
    filename = "square-circle-coarse-o2"
    domain, Fn = ConstructDomain(porder, θ, filename)
    ngp = Int64(sqrt(length(domain.elements[1].weights)))
    
    N_x, _ = size(inputs)
    L = 1
    xx = LinRange(0, L, N_x)
    ax[1,i].plot(xx, inputs[:, ind], "--o",color="#808080", fillstyle="none")
    ax[1,i].set_title(nns[i],pad = 5)
    ax[1,i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[1,i].set_xticks([])
    ax[1,i].set_ylim([-300,400])
    
    # vmin, vmax = minimum(outputs[:, ind]), maximum(outputs[:, ind])
    vmin = 0
    vmax = 350
    visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind],     ax = ax[2,i], mycolorbar=false )
    im3 = visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind + 3], ax = ax[3,i], mycolorbar=false )

    ax[1,i].spines["top"].set_visible(false)
    ax[1,i].spines["right"].set_visible(false)
    ax[1,i].spines["left"].set_color("#808080")
    ax[1,i].spines["left"].set_linewidth(0.3)
    ax[1,i].spines["bottom"].set_color("#808080")
    ax[1,i].spines["bottom"].set_linewidth(0.3)
    if i > 1
        ax[1,i].set_yticklabels([])
    end

    for j = 2:3
        ax[j,i].spines["top"].set_visible(false)
        ax[j,i].spines["right"].set_visible(false)
        ax[j,i].spines["left"].set_visible(false)
        ax[j,i].spines["bottom"].set_visible(false)
        ax[j,i].set_aspect("equal","box")
        ax[j,i].set_yticks([])
        ax[j,i].set_xticks([])
    end

    if i == 4
        cax2 = fig.add_axes([0.92,0.025, 0.015, 0.575])
        cb2 = plt.colorbar(im3,cax=cax2,ticks=[0,50,100,150,200,250,300,350])
        cb2.outline.set_visible(false)
        cb2.ax.yaxis.set_tick_params(colors="#808080",width=0.3)
    end
end
ax[1,1].set_ylabel(L"\tau",labelpad=5)
ax[2,1].set_ylabel("True displacement",labelpad=5)
ax[3,1].set_ylabel("Predicted displacement",labelpad=5)
plt.subplots_adjust(left = 0.05, right = 0.9, bottom = 0.025,top=.9,hspace=0.1,wspace=0.1)
fig.savefig("Solid-worst.pdf")