using NPZ
using LinearAlgebra
using PyPlot
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 15
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
merge!(rcParams, font0)

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
    
end


map_plot()
prediction_plot("PCA", 10000, 128, 1)
prediction_plot("PCA", 10000, 128, 2)
prediction_plot("PCA", 10000, 128, 3)

prediction_plot("FNO", 10000, 16, 1)
prediction_plot("FNO", 10000, 16, 2)
prediction_plot("FNO", 10000, 16, 3)


prediction_plot("DeepONet", 10000, 128, 1)
prediction_plot("DeepONet", 10000, 128, 2)
prediction_plot("DeepONet", 10000, 128, 3)


prediction_plot("PARA", 10000, 128, 1)
prediction_plot("PARA", 10000, 128, 2)
prediction_plot("PARA", 10000, 128, 3)

# 



