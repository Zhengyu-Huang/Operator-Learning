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
    N_x = length(data)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig = figure()
    plot(xx, data, "--o", fillstyle="none", color="C0")
    fig.tight_layout()
    fig.savefig(file_name)
end


function output_plot(data, file_name)
    N_x = length(data)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig = figure()
    plot(xx, data, "--o", fillstyle="none", color="C1")
    fig.tight_layout()
    fig.savefig(file_name)
end

function map_plot(prefix = "../src/", inds = [1,11])
    inputs   = npzread(prefix * "adv_a0.npy")   
    outputs  = npzread(prefix * "adv_aT.npy")
    
    ######################################################
    N_x, N_data = size(outputs)

    L = 1
    xx = LinRange(0, L, N_x)

    for ind in inds
        input_plot(inputs[:, ind], "Advection-low-map-input-$(ind).pdf")
        output_plot(outputs[:, ind], "Advection-low-map-output-$(ind).pdf")
    end
end



# i = 0, 1, 2 smallest, median, largest
function prediction_plot(nn_name, ntrain, width, ind)
    err = ["s", "m", "l"]
    inputfile = nn_name * "/" * string(ntrain) * "_" * string(width) * "_test_input_save.npy"
    outputfile = nn_name * "/" * string(ntrain) * "_" * string(width) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    
    N_x, _ = size(inputs)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig, ax = PyPlot.subplots(ncols = 3, sharex=true, sharey=true, figsize=(18,6))
    
    ax[1].plot(xx, inputs[:, ind], "--o", fillstyle="none", color="C0")
    ax[2].plot(xx, outputs[:, ind], "--o", fillstyle="none", color="C1")
    ax[3].plot(xx, outputs[:, ind+3], "--o", fillstyle="none", color="C2")
    
    @info "error is: ", norm(outputs[:, ind+3] - outputs[:, ind])/norm(outputs[:, ind])
    
    fig.tight_layout()
    fig.savefig("Advection-low-" * err[ind] * "_" * nn_name * "_" * string(ntrain) * "_" * string(width)* ".pdf")
    
end


map_plot("../src/")
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