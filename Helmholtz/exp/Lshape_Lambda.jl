# using Helmholtz

include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

function Lshape_Dirichlet_To_Neumann(params::Array{Float64, 1}, nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
    """
    -Δu - ω^2/c^2 u = f  in Ω=[0,1]×[0,1]
    
    boundary condition orders:
    
    bottom N = (0, -1)  
    right  N = (1,  0)  
    top    N = (0,  1) 
    left   N = (-1, 0)  
    
    """
    
    """
    ---5---
    |     |4      
    |     |      
    6     ___3__    
    |           |
    |           |2
    ----- 1 ----
    """
    bc1, bc2, bc3, bc4, bc5, bc6, c = params
    bc_types = ["Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet"]
    bc_funcs = [(x,y)-> bc1*sin(π*x),  (x,y)-> bc2*sin(2π*y),  (x,y)-> bc3*sin(2π*(x-0.5)),  (x,y)-> bc4*sin(2π*(y-0.5)), (x,y)-> bc5*sin(2π*x),  (x,y)-> bc6*sin(π*y)]
    ω = 1000.0
    c_func = (x,y)->  c
    

    s_func = (x,y)-> 1
    
    Lx, Ly = 1.0, 1.0
    nodes, elnodes, bc_nodes = lshape(Lx, Ly, nex, ney, porder)
    
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
    visScalarField(domain, domain.state, savefile="Lshape.png")
    
    DBC_∂u∂n_ele = computeNeumannOnDirichletEdge(domain)
    ∂u∂n_data = computeNeumannOnDirichletNode(domain, DBC_∂u∂n_ele, bc_nodes, bc_types)
    
    fig_disp, ax_disp = PyPlot.subplots(ncols = 6, nrows=1, sharex=false, sharey=true, figsize=(16,4))
    markersize = 1
    for bc_id = 1:6
        bc_id_nodes = bc_nodes[:, bc_id]
        xx, yy = nodes[bc_id_nodes, 1], nodes[bc_id_nodes, 2]
        data_Dirichlet = [bc_funcs[bc_id](xx[i], yy[i]) for i =1:length(xx)]
        data_Neumann = ∂u∂n_data[bc_id, bc_id_nodes]
        if bc_id == 1 || bc_id == 3 || bc_id == 5
            ax_disp[bc_id].plot(xx, data_Dirichlet, "ro", markersize = markersize, label="Dirichlet")
            ax_disp[bc_id].plot(xx, data_Neumann, "go", markersize = markersize, label="Neumann")
            ax_disp[bc_id].set_xlabel("x")
        else
            ax_disp[bc_id].plot(yy, data_Dirichlet, "ro", markersize = markersize, label="Dirichlet")
            ax_disp[bc_id].plot(yy, data_Neumann, "go", markersize = markersize, label="Neumann")
            ax_disp[bc_id].set_xlabel("y")
        end
        ax_disp[bc_id].grid()
    end 
    ax_disp[1].legend()
    fig_disp.tight_layout()
end

nex = 80 
ney = 80
porder = 2
ngp = 3
params = [1.0;2.0;3.0;4.0;5.0;6.0; 334]
Lshape_Dirichlet_To_Neumann(params, nex, ney, porder, ngp)