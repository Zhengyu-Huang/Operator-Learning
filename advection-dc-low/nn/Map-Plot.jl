using NPZ
using LinearAlgebra
using PyPlot
include("../../plotdefaults.jl")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
font0 = Dict(
    "lines.markersize" =>0.5,
    )
merge!(rcParams, font0)

function meshgrid(xin, yin)
  return  xin' .* ones(length(yin)) , ones(length(xin))' .* yin
end


function input_plot(data, file_name)
    N_x = length(data)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig = figure(figsize=(2,1.5))
    plot(xx, data, "--o", fillstyle="none",markersize=1, color="#a1a1a1")
    ax = gca()
    ax.set_title(L"a_0",pad = 2)   
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["left"].set_color("#808080")
    ax.spines["left"].set_linewidth(0.3)
    ax.spines["bottom"].set_color("#808080")
    ax.spines["bottom"].set_linewidth(0.3)
    ax[:set_xlim]([0,1])
    ax.set_xlabel(L"x",labelpad=1)
    ax[:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[:yaxis][:set_tick_params](colors="#808080",width=0.3)
    plt.subplots_adjust(bottom = 0.2,top=.85,left=0.15)

    fig.savefig(file_name)
end


function output_plot(data, file_name)
    N_x = length(data)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig = figure(figsize=(2,1.5))
    plot(xx, data, "--o", fillstyle="none",markersize=1, color="#a1a1a1")
    ax = gca()
    ax.set_title(L"a(T)",pad = 2)   
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["left"].set_color("#808080")
    ax.spines["left"].set_linewidth(0.3)
    ax.spines["bottom"].set_color("#808080")
    ax.spines["bottom"].set_linewidth(0.3)
    ax[:set_xlim]([0,1])
    ax.set_xlabel(L"x",labelpad=1)
    ax[:xaxis][:set_tick_params](colors="#808080",width=0.3)
    ax[:yaxis][:set_tick_params](colors="#808080",width=0.3)
    plt.subplots_adjust(bottom = 0.2,top=.85,left=0.15)
    fig.savefig(file_name)
end

function map_plot(prefix = "../../data/", inds = [18,28])
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
    for i =1:3
        ax[i].spines["top"].set_visible(false)
        ax[i].spines["right"].set_visible(false)
        ax[i].spines["left"].set_color("#808080")
        ax[i].spines["bottom"].set_color("#808080")
    end
    
    @info "error is: ", norm(outputs[:, ind+3] - outputs[:, ind])/norm(outputs[:, ind])
    
    fig.tight_layout()
    fig.savefig("Advection-low-" * err[ind] * "_" * nn_name * "_" * string(ntrain) * "_" * string(width)* ".pdf")
    
end

nn_names = ["PCA", "DeepONet", "PARA", "FNO"]
map_plot("../src/")