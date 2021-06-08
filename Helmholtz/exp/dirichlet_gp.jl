# using Helmholtz

include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

function D2N(ind::Int64, xi::Array{Float64, 1}, c::Float64, nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
    """
    -Δu - ω^2/c^2 u = f  in Ω=[0,1]×[0,1]
    
    boundary condition orders:
    
    bottom N = (0, -1)  
    right  N = (1,  0)  
    top    N = (0,  1) 
    left   N = (-1, 0)  
    
    """
    
    """
    ----- 3 -----
    |           |
    |           |
    4           2
    |           |
    |           |
    ----- 1 -----
    """
    bc_types = ["Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet"]
    
    # define fourier boundary
    fl = (x,l)-> (l*π)^-2 * xi[l] * sin.(l*π*x)
    f = x-> sum(l->fl(x,l),1:length(xi))

    bc_funcs = [(x,y)->0, (x,y)->0, (x,y)->f(x), (x,y)->0]
    ω = 1000.0
    c_func = (x,y)->  c
    s_func = (x,y)-> 0 
    
    Lx, Ly = 1.0, 1.0
    nodes, elnodes, bc_nodes = box(Lx, Ly, nex, ney, porder)
    
    @info "construct domain"
    domain = Domain(nodes, elnodes,
    bc_nodes, bc_types, bc_funcs,
    porder, ngp; 
    ω = ω, 
    c_func = c_func, 
    s_func = s_func)
    
    @info "linear solve"
    domain = solve!(domain)
    
    @info "start visualize"
    visScalarField(domain, domain.state, savefile="sinBox"*string(ind)*".png")
    
    DBC_∂u∂n_ele = computeNeumannOnDirichletEdge(domain)
    ∂u∂n_data = computeNeumannOnDirichletNode(domain, DBC_∂u∂n_ele, bc_nodes, bc_types)
    
    # fig_disp, ax_disp = PyPlot.subplots(ncols = 4, nrows=1, sharex=false, sharey=true, figsize=(16,4))
    # markersize = 1
    @show size(∂u∂n_data)
    for bc_id = 1:4
        bc_id_nodes = bc_nodes[:, bc_id]
        xx, yy = nodes[bc_id_nodes, 1], nodes[bc_id_nodes, 2]
        data_Dirichlet = [bc_funcs[bc_id](xx[i], yy[i]) for i =1:length(xx)]
        data_Neumann = ∂u∂n_data[bc_id, bc_id_nodes]
        @show size(data_Neumann)
        # if bc_id == 1 || bc_id == 3
        #     ax_disp[bc_id].plot(xx, data_Dirichlet, "ro", markersize = markersize, label="Dirichlet")
        #     ax_disp[bc_id].plot(xx, data_Neumann, "go", markersize = markersize, label="Neumann")
        #     ax_disp[bc_id].set_xlabel("x")
        # else
        #     ax_disp[bc_id].plot(yy, data_Dirichlet, "ro", markersize = markersize, label="Dirichlet")
        #     ax_disp[bc_id].plot(yy, data_Neumann, "go", markersize = markersize, label="Neumann")
        #     ax_disp[bc_id].set_xlabel("y")
        # end
        # ax_disp[bc_id].grid()
    end 
    # ax_disp[1].legend()
    # fig_disp.tight_layout()
end


n_data = 100
max_l = 100

# wave speed
c = 500.

# domain discretization set-up
Lx, Ly = 1.0, 1.0   # box edge lengths
nex, ney = 20, 20   # grid size
porder = 2          # FE polynomial order
ngp = 3             # num quad points
nodes, elnodes, bc_nodes = box(Lx, Ly, nex, ney, porder)

# source, c(x), ω definitions
c_func = (x,y)-> c
s_func = (x,y)-> 0 
ω = 1000.0

# bc type
bc_types = ["Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet"]

# set-up boundary data storage
top_nodes = bc_nodes[:, 3]
top_x, top_y = nodes[top_nodes, 1], nodes[top_nodes, 2]
side_nodes = bc_nodes[:, 2]
side_x, side_y = nodes[side_nodes, 1], nodes[side_nodes, 2]

f_snaps = zeros(length(top_x),n_data)
g_snaps = zeros(2*length(top_x)+2*length(side_x),n_data)

for i = 1:n_data
    # define sinusoidal GP for boundary 3
    xi = randn(max_l)
    fl = (x,l)-> (l*π)^-2 * xi[l] * sin.(l*π*x)
    f = x-> sum(l->fl(x,l),1:length(xi))
    bc_funcs = [(x,y)->0, (x,y)->0, (x,y)->f(x), (x,y)->0]

    # @info "construct domain"
    domain = Domain(nodes, elnodes, bc_nodes, bc_types, bc_funcs, porder, ngp; 
    ω = ω, c_func = c_func, s_func = s_func)
    
    # @info "linear solve"
    domain = solve!(domain)

    # @info "get Neumann data"
    DBC_∂u∂n_ele = computeNeumannOnDirichletEdge(domain)
    ∂u∂n_data = computeNeumannOnDirichletNode(domain, DBC_∂u∂n_ele, bc_nodes, bc_types)

    f_snaps[:,i] = [bc_funcs[3](top_x[j],top_y[j]) for j = 1:length(top_x)]

    for bc_id = 1:4
        bc_id_nodes = bc_nodes[:, bc_id]
        data_Neumann = ∂u∂n_data[bc_id, bc_id_nodes]
        g_snaps[(bc_id-1)*length(top_x)+1:bc_id*length(top_x),i] = data_Neumann
    end
end

# generate test set
n_test = 1000
f_test = zeros(length(top_x),n_test)
g_test = zeros(2*length(top_x)+2*length(side_x),n_test)

for i = 1:n_test
    # define sinusoidal GP for boundary 3
    xi = randn(max_l)
    fl = (x,l)-> (l*π)^-2 * xi[l] * sin.(l*π*x)
    f = x-> sum(l->fl(x,l),1:length(xi))
    bc_funcs = [(x,y)->0, (x,y)->0, (x,y)->f(x), (x,y)->0]

    # @info "construct domain"
    domain = Domain(nodes, elnodes, bc_nodes, bc_types, bc_funcs, porder, ngp; 
    ω = ω, c_func = c_func, s_func = s_func)
    
    # @info "linear solve"
    domain = solve!(domain)

    # @info "get Neumann data"
    DBC_∂u∂n_ele = computeNeumannOnDirichletEdge(domain)
    ∂u∂n_data = computeNeumannOnDirichletNode(domain, DBC_∂u∂n_ele, bc_nodes, bc_types)

    f_test[:,i] = [bc_funcs[3](top_x[j],top_y[j]) for j = 1:length(top_x)]

    for bc_id = 1:4
        bc_id_nodes = bc_nodes[:, bc_id]
        data_Neumann = ∂u∂n_data[bc_id, bc_id_nodes]
        g_test[(bc_id-1)*length(top_x)+1:bc_id*length(top_x),i] = data_Neumann
    end
end

U_f,s_f,V_f = svd(f_snaps)
U_g,s_g,V_g = svd(g_snaps)

PyPlot.figure()
PyPlot.plot(top_x,f_snaps[:,1:10])
PyPlot.xlabel("x")
PyPlot.ylabel("f")
PyPlot.title("Sin GP Dirichlet data on top boundary")
PyPlot.savefig("../figs/bc3_examples_"*string(c)*".png")

PyPlot.figure()
PyPlot.plot(top_x,U_f[:,1:5])
PyPlot.xlabel("x")
PyPlot.ylabel("u")
PyPlot.title("leading PCA basis functions for Dirichlet data f")
PyPlot.savefig("../figs/f_basis_"*string(c)*".png")

m = 41
PyPlot.figure()
PyPlot.subplot(2,2,1)
PyPlot.plot(top_x,U_g[1:m,1:5])
PyPlot.title("bottom Neumann PCA")
PyPlot.subplot(2,2,2)
PyPlot.plot(side_y,U_g[m+1:2*m,1:5])
PyPlot.title("right Neumann PCA")
PyPlot.subplot(2,2,3)
PyPlot.plot(top_x,U_g[2*m+1:3*m,1:5])
PyPlot.title("top Neumann PCA")
PyPlot.subplot(2,2,4)
PyPlot.plot(side_y,U_g[3*m+1:4*m,1:5])
PyPlot.title("left Neumann PCA")
PyPlot.savefig("../figs/g_basis_"*string(c)*".png")

PyPlot.figure()
PyPlot.subplot(1,2,1)
PyPlot.semilogy(s_f/s_f[1])
PyPlot.title("top Dirichlet data spectrum")
PyPlot.subplot(1,2,2)
PyPlot.semilogy(s_g/s_g[1])
PyPlot.title("Neumann data spectrum")
PyPlot.savefig("../figs/PCA_spectrum_"*string(c)*".png")


errs = zeros(20,2)
for r = 1:20
    rf = r
    rg = r

    # learn linear operator
    D = transpose(f_snaps)*U_f[:,1:rf]
    R = transpose(g_snaps)*U_g[:,1:rg]
    L = transpose(D\R)

    # compute training err
    err = U_g[:,1:rg]*L*transpose(D) - g_snaps
    rel_sq_err = sum(err.^2,dims=1)./sum(g_snaps.^2,dims=1)
    mrse = sum(rel_sq_err)/n_data

    # compute test err
    D_test = transpose(f_test)*U_f[:,1:rf]
    err_test = U_g[:,1:rg]*L*transpose(D_test) - g_test
    rel_sq_err_test = sum(err_test.^2,dims=1)./sum(g_test.^2,dims=1)
    mrse_test = sum(rel_sq_err_test)/n_test

    errs[r,:] = [mrse, mrse_test]
end


PyPlot.figure()
PyPlot.semilogy(errs)
PyPlot.legend(["training","test"])
PyPlot.xlabel("Num. PCA basis functions")
PyPlot.ylabel("Mean rel sq error")
PyPlot.savefig("../figs/op_errs_"*string(c)*".png")

PyPlot.close("all")

