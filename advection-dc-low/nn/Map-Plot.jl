using NPZ
using LinearAlgebra
using PyPlot
include("../../plotdefaults.jl")


function meshgrid(xin, yin)
  return  xin' .* ones(length(yin)) , ones(length(xin))' .* yin
end


function input_plot(data, file_name)
    N_x = length(data)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig = figure()
    plot(xx, data, "--o", fillstyle="none",markersize=6, color="#a1a1a1")
    ax = gca()
    ax.set_title(L"a_0",pad = 20)   
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["left"].set_color("#808080")
    ax.spines["bottom"].set_color("#808080")
    ax[:set_xlim]([0,1])
    ax.set_xlabel(L"x",labelpad=10)
    ax[:xaxis][:set_tick_params](colors="#808080")
    ax[:yaxis][:set_tick_params](colors="#808080")
    plt.subplots_adjust(bottom = 0.2,top=.85)

    fig.savefig(file_name)
end


function output_plot(data, file_name)
    N_x = length(data)
    L = 1
    xx = LinRange(0, L, N_x)
    
    fig = figure()
    plot(xx, data, "--o", fillstyle="none",markersize=6, color="#a1a1a1")
    ax = gca()
    ax.set_title(L"a(T)",pad = 20)   
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["left"].set_color("#808080")
    ax.spines["bottom"].set_color("#808080")
    ax[:set_xlim]([0,1])
    ax.set_xlabel(L"x",labelpad=10)
    ax[:xaxis][:set_tick_params](colors="#808080")
    ax[:yaxis][:set_tick_params](colors="#808080")
    plt.subplots_adjust(bottom = 0.2,top=.85)
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


nn_names = ["PCA", "DeepONet", "PARA", "FNO"]
map_plot("../src/")


ntrain = 10000
widths = [128, 128, 128, 16]
ind = 2 # median error
fig, ax = PyPlot.subplots(3,4, sharex=true, sharey=true, figsize=(6.5,3.25))
for i = 1:4
    nn_name = nn_names[i]
    inputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
    outputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    
    N_x, _ = size(inputs)
    L = 1
    xx = LinRange(0, L, N_x)

    ax[1,i].plot(xx, inputs[:, ind], "--o", fillstyle="none", color="#a1a1a1",markersize=1)
    ax[2,i].plot(xx, outputs[:, ind], "--o", fillstyle="none",color="#a1a1a1",markersize=1)
    ax[3,i].plot(xx, outputs[:, ind+3], "--o", fillstyle="none",color=colors[i],markersize=1)

    ax[1,i].set_title(nns[i],pad = 2)
    ax[3,i].set_xlabel(L"x",labelpad=1)

    for j = 1:3
        ax[j,i].spines["top"].set_visible(false)
        ax[j,i].spines["right"].set_visible(false)
        ax[j,i].spines["left"].set_color("#808080")
        ax[j,i].spines["left"].set_linewidth(0.3)
        ax[j,i].spines["bottom"].set_color("#808080")
        ax[j,i].spines["bottom"].set_linewidth(0.3)
        ax[j,i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
        ax[j,i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
    end
end
ax[1,1].set_ylabel(L"a_0")
ax[2,1].set_ylabel("True "*L"a(T)")
ax[3,1].set_ylabel("Predicted "*L"a(T)")
plt.subplots_adjust(left = 0.08, right = 0.98, bottom = 0.1,top=.9,hspace=0.1,wspace=0.1)
plt.savefig("Advection-dc-low-medians.pdf")

ind = 3 # largest error
fig, ax = PyPlot.subplots(3,4, sharex=true, sharey=true, figsize=(6.5,3.25))
for i = 1:4
    nn_name = nn_names[i]
    inputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_input_save.npy"
    outputfile = nn_name * "/" * string(ntrain) * "_" * string(widths[i]) * "_test_output_save.npy"
    inputs   = npzread(inputfile)   
    outputs  = npzread(outputfile)
    
    N_x, _ = size(inputs)
    L = 1
    xx = LinRange(0, L, N_x)

    ax[1,i].plot(xx, inputs[:, ind], "--o", fillstyle="none",color="#a1a1a1",markersize=1)
    ax[2,i].plot(xx, outputs[:, ind], "--o", fillstyle="none",color="#a1a1a1",markersize=1)
    ax[3,i].plot(xx, outputs[:, ind+3], "--o", fillstyle="none", color=colors[i],clip_on=false,markersize=1)

    ax[1,i].set_title(nns[i],pad = 2)
    ax[3,i].set_xlabel(L"x",labelpad=1)

    for j = 1:3
        ax[j,i].spines["top"].set_visible(false)
        ax[j,i].spines["right"].set_visible(false)
        ax[j,i].spines["left"].set_color("#808080")
        ax[j,i].spines["left"].set_linewidth(0.3)
        ax[j,i].spines["bottom"].set_color("#808080")
        ax[j,i].spines["bottom"].set_linewidth(0.3)
        ax[j,i][:xaxis][:set_tick_params](colors="#808080",width=0.3)
        ax[j,i][:yaxis][:set_tick_params](colors="#808080",width=0.3)
        ax[j,i].set_ylim([-1.3,1.3])
    end
end
ax[1,1].set_ylabel(L"a_0")
ax[2,1].set_ylabel("True "*L"a(T)")
ax[3,1].set_ylabel("Predicted "*L"a(T)")
# plt.tight_layout()
plt.subplots_adjust(left = 0.08, right = 0.98, bottom = 0.1,top=.9,hspace=0.1,wspace=0.1)
plt.savefig("Advection-dc-low-worst.pdf")