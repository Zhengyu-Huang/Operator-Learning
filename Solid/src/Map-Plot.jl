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

function quad_itp(x,y)
    n = trunc(Int,(length(y)-1)/2) # number of segments
    xx = LinRange(x[1],x[end],n*10+1)
    yy = zeros(n*10+1,1)
    for i = 1:n
        xp = x[2*i-1:2*i+1]
        X = [xp.^2 xp ones(3,1)]
        a = X\y[2*i-1:2*i+1]
        yy[10*(i-1)+1:10*i] = a[1].*xx[10*(i-1)+1:10*i].^2 + a[2].*xx[10*(i-1)+1:10*i] .+ a[3]
    end
    yy[end] = y[end]
    return xx,yy
end
# map_plot()



ntrain = 10000
widths = [128, 128, 128, 16]

for ind = 2:3 # medians/worst-case

fig2,ax2 = PyPlot.subplots(4,4,sharex=true,figsize = (6.5,5))
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
    px,py = quad_itp(xx,inputs[:,ind])

    ax2[1,i].plot(xx, inputs[:, ind], "o",color="#808080",markersize=1, fillstyle="none")
    ax2[1,i].plot(px,py,color="#808080")
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
    ax2[1,i].set_aspect(1. /900,anchor="S")

    vmin = 0
    vmax = 350
    @show minimum(minimum([outputs[:,ind],outputs[:,ind+3]]))
    @show maximum(maximum([outputs[:,ind],outputs[:,ind+3]]))

    @show minimum(broadcast(abs,outputs[:,ind]-outputs[:,ind+3]))
    @show maximum(broadcast(abs,outputs[:,ind]-outputs[:,ind+3]))
    visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind],     ax = ax2[2,i])
    im3 = visσ(domain, ngp, vmin, vmax; σ=outputs[:, ind + 3], ax = ax2[3,i])
    im4 = visσ(domain, ngp, 0, 90; σ=broadcast(abs,(outputs[:, ind + 3]-outputs[:,ind])), ax = ax2[4,i], mycolorbar="magma" )

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
        cax2 = fig2.add_axes([0.92,0.2693, 0.015, 0.46638])
        cb2 = plt.colorbar(im3,cax=cax2,ticks=[0,50,100,150,200,250,300,350])
        cb2.outline.set_visible(false)
        cb2.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

        cax = fig2.add_axes([0.92,0.025, 0.015, 0.222])
        cb = plt.colorbar(im4,cax=cax,ticks=[0, 30,60, 90])
        cb.outline.set_visible(false)
        cb.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

    end
end
ax2[1,1].set_ylabel("Top loading "*L"\bar{t}(x)",labelpad=1)
ax2[2,1].set_ylabel("True stress field",labelpad=26)
ax2[3,1].set_ylabel("Predicted stress field",labelpad=26)
ax2[4,1].set_ylabel("Stress field error",labelpad=26)
fig2.subplots_adjust(left = 0.08, right = 0.9, bottom = 0.025,top=0.98,hspace=0.1,wspace=0.1)
if ind==2
    fig2.savefig("Solid-medians.pdf")
elseif ind==3
    fig2.savefig("Solid-worst.pdf")
end
end