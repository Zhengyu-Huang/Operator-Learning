# using Helmholtz

include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

function Ring_Dirichlet_To_Neumann(params::Array{Float64, 1}, neθ::Int64 = 20, ner::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
    """
    -Δu - ω^2/c^2 u = f  in Ω=[0,2π]×[1,2]
    
    boundary condition orders:
    
    inner   N = -(x, y)/sqrt(x^2 + y^2)
    outter  N = (x, y)/sqrt(x^2 + y^2)
    
        ****2****
      *           * 
    *    *  *  *    * 
    *    *  1  *    * 
    *    *  *  *   *
      *           *
        *********
    """
    bc1, bc2, c = params
    bc_types = ["Dirichlet", "Dirichlet"]

    # (x,y)-> bc1*sin(θ),  (x,y)-> bc2*sin(θ)
    bc_funcs = [(x,y)-> bc1*y/sqrt(x^2 + y^2),  (x,y)-> bc2*y/sqrt(x^2 + y^2)]
    ω = 0.0
    c_func = (x,y)->  c
    s_func = (x,y)-> 1
    

    r, R = 1.0, 2.0
    nodes, elnodes, bc_nodes = ring(r, R, neθ, ner, porder)
    
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
    visScalarField(domain, domain.state, savefile="Ring.png")
    
    DBC_∂u∂n_ele = computeNeumannOnDirichletEdge(domain)
    ∂u∂n_data = computeNeumannOnDirichletNode(domain, DBC_∂u∂n_ele, bc_nodes, bc_types)
    
    fig_disp, ax_disp = PyPlot.subplots(ncols = 2, nrows=1, sharex=false, sharey=true, figsize=(8,4))
    markersize = 1
    for bc_id = 1:2
        bc_id_nodes = bc_nodes[:, bc_id]
        xx, yy = nodes[bc_id_nodes, 1], nodes[bc_id_nodes, 2]
        data_Dirichlet = [bc_funcs[bc_id](xx[i], yy[i]) for i =1:length(xx)]
        data_Neumann = ∂u∂n_data[bc_id, bc_id_nodes]

        ax_disp[bc_id].plot(atan.(yy,xx), data_Dirichlet, "ro", markersize = markersize, label="Dirichlet")
        ax_disp[bc_id].plot(atan.(yy,xx), data_Neumann, "go", markersize = markersize, label="Neumann")
        ax_disp[bc_id].set_xlabel("θ")
        ax_disp[bc_id].grid()

    end 
    ax_disp[1].legend()
    fig_disp.tight_layout()
end

neθ = 40
ner = 20
porder = 2
ngp = 3
params = [1.0;2.0; 334]
Ring_Dirichlet_To_Neumann(params, neθ, ner, porder, ngp)