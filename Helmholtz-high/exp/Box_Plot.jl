# using Helmholtz

include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

function Box_Dirichlet_To_Neumann(c::Float64, bc3::Function, file_pre::String, nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
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
    bc_funcs = [(x,y)-> 0,  (x,y)-> 0,  bc3,  (x,y)-> 0]
    ω = 1000.0
    c_func = (x,y)->  c
    
    x_0, y_0 = 0.3, 0.5
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
    
    visScalarField(domain, domain.state, savefile = file_pre*"Box$(c).png")
    PyPlot.close("all")


    DBC_∂u∂n_ele = computeNeumannOnDirichletEdge(domain)
    ∂u∂n_data = computeNeumannOnDirichletNode(domain, DBC_∂u∂n_ele, bc_nodes, bc_types)
    
    fig_disp, ax_disp = PyPlot.subplots(ncols = 4, nrows=1, sharex=false, sharey=false, figsize=(16,4))
    markersize = 1
    for bc_id = 1:4
        bc_id_nodes = bc_nodes[:, bc_id]
        xx, yy = nodes[bc_id_nodes, 1], nodes[bc_id_nodes, 2]
        data_Dirichlet = [bc_funcs[bc_id](xx[i], yy[i]) for i =1:length(xx)]
        data_Neumann = ∂u∂n_data[bc_id, bc_id_nodes]
        if bc_id == 1 || bc_id == 3
            ax_disp[bc_id].plot(xx, data_Dirichlet, "ro", markersize = markersize, label="Dirichlet")
            ax_disp[bc_id].plot(xx, data_Neumann, "-go", markersize = markersize, label="Neumann")
            ax_disp[bc_id].set_xlabel("x")
        else
            ax_disp[bc_id].plot(yy, data_Dirichlet, "ro", markersize = markersize, label="Dirichlet")
            ax_disp[bc_id].plot(yy, data_Neumann, "-go", markersize = markersize, label="Neumann")
            ax_disp[bc_id].set_xlabel("y")
        end
        ax_disp[bc_id].grid()
    end 
    ax_disp[1].legend()
    fig_disp.tight_layout()
    fig_disp.savefig(file_pre*"D2N$(c).png")
    PyPlot.close("all")
end

nex = 100 
ney = 100
porder = 2
ngp = 3

A, σ, μ = 1.0, 0.1, 0.5
c, bc3 = 10.0, (x, y) -> A/(σ *sqrt(2*pi)) * exp(-1/(2*σ^2) * (x - μ)^2)
file_pre = "Gaussian-1" # 1-> μ=0.5  2-> μ=0.3
Box_Dirichlet_To_Neumann(c, bc3, file_pre, nex, ney, porder, ngp)



# l = 4
# c, bc3 = 100.0, (x, y) -> (l*π)^-2 * sin(l*π*x)
# file_pre = "Sine-$(l)" # 1-> μ=0.5  2-> μ=0.3
# Box_Dirichlet_To_Neumann(c, bc3, file_pre, nex, ney, porder, ngp)