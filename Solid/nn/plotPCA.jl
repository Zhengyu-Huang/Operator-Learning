using NPZ
using LinearAlgebra
using PyPlot
include("../../plotdefaults.jl")

using Random, Distributions
include("../src/PlatePull.jl")


DeepONetPCA_data = npzread("DeepONet/DeepONetPCA_data.npy")
Ug = npzread("PCA/PCA_data.npy")
    

porder = 2
θ = rand(Normal(0, 1.0), 100);
filename = "../src/square-circle-coarse-o2"
domain, Fn = ConstructDomain(porder, θ, filename)
ngp = Int64(sqrt(length(domain.elements[1].weights)))

# fig2,ax = PyPlot.subplots(3,4,sharex=true,figsize = (6.5,5))
fig2 = PyPlot.figure(figsize=(6,5))
subfigs = fig2.subfigures(nrows=3,ncols=1)

ax1 = subfigs[1].subplots(1,4)
ax2 = subfigs[2].subplots(1,4)
ax3 = subfigs[3].subplots(1,4)
ax = [ax1, ax2, ax3]
ims = Array{Any}(undef,3,4)
for i = 1:4
    ims[1,i] = visσ(domain, ngp, -0.1, 0.1; σ=Ug[:, i], ax = ax1[i], mycolorbar="gray")
    ims[2,i] = visσ(domain, ngp,0,0.15; σ=DeepONetPCA_data[:, 2*i-1], ax = ax2[i], mycolorbar="gray")
    ims[3,i] = visσ(domain, ngp,-0.05,0.1; σ=DeepONetPCA_data[:, 2*i], ax = ax3[i], mycolorbar="gray")
    # @show minimum(DeepONetPCA_data[:, 2*i]),maximum(DeepONetPCA_data[:, 2*i])

    for j = 1:3
        ax[j][i].spines["top"].set_visible(false)
        ax[j][i].spines["right"].set_visible(false)
        ax[j][i].spines["left"].set_visible(false)
        ax[j][i].spines["bottom"].set_visible(false)
        ax[j][i].set_aspect("equal", "box")
        ax[j][i].set_yticks([])
        ax[j][i].set_xticks([])
    end
end

subfigs[1].suptitle("Leading PCA basis functions",fontsize=14,y=1)
subfigs[2].suptitle("Trained DeepONet trunk functions",fontsize=14,y=1)
subfigs[3].suptitle("Leading PCA modes of trained DeepONet trunk functions",fontsize=14,y=1)
plt.subplots_adjust(left = 0.01, right = 0.9, bottom = 0.025,top=.95,hspace=0.2,wspace=0.1)

i = 4
temp = ax[1][i].get_position()
xw = temp.x1-temp.x0
cax = subfigs[1].add_axes([temp.x1+0.03*xw, temp.y0, 0.07*xw, temp.y1-temp.y0],frameon=false)
cb = plt.colorbar(ims[1,i],cax=cax)
cb.outline.set_visible(false)
cb.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

temp = ax[2][i].get_position()
xw = temp.x1-temp.x0
cax = subfigs[2].add_axes([temp.x1+0.03*xw, temp.y0, 0.07*xw, temp.y1-temp.y0],frameon=false)
cb = plt.colorbar(ims[2,i],cax=cax)
cb.outline.set_visible(false)
cb.ax.yaxis.set_tick_params(colors="#808080",width=0.3)

temp = ax[3][i].get_position()
xw = temp.x1-temp.x0
cax = subfigs[3].add_axes([temp.x1+0.03*xw, temp.y0, 0.07*xw, temp.y1-temp.y0],frameon=false)
cb = plt.colorbar(ims[3,i],cax=cax)
cb.outline.set_visible(false)
cb.ax.yaxis.set_tick_params(colors="#808080",width=0.3)


fig2.savefig("Solid-pca-vis.pdf")
plt.close()