using NPZ
using LinearAlgebra
using PyPlot

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 16
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
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




map_plot("../../data/")
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

# prediction_plot("DeepONet", 10000, 128)
# prediction_plot("FNO", 10000, 16)