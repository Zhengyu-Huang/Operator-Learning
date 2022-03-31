# using Helmholtz

include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

function Box(nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
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
    bc_types = ["Dirichlet", "Dirichlet"]
    bc_funcs = [(x,y)-> x^2,  (x,y)-> x^2]
    uref_func = (x,y)-> x^2
    ω = 0.0
    c_func = (x,y)-> 1
    s_func = (x,y)-> -2

    # bc_funcs = [(x,y)-> x+y,  (x,y)-> x+y]
    # uref_func = (x,y)-> x+y
    # ω = 0.0
    # c_func = (x,y)-> 1
    # s_func = (x,y)-> 0
    
    Lx, Ly = 1.0, 2.0
    nodes, elnodes, bc_nodes = ring(Lx, Ly, nex, ney, porder)
    
    @info "construct domain"
    domain = Domain(nodes, elnodes,
    bc_nodes, bc_types, bc_funcs,
    porder, ngp; 
    ω = ω, 
    c_func = c_func, 
    s_func = s_func)
    
    @info "linear solve"
    domain = solve!(domain)
    
    state_ref = copy(domain.state)
    for i = 1:domain.nnodes
        state_ref[i] = uref_func(domain.nodes[i, :]...)
    end

    @info "start visualize"
    @info domain.state
    @info state_ref
    
    visScalarField(domain, domain.state - state_ref, savefile="Box.png")
end

nex = 20
ney = 8
porder = 1
ngp = 3
Box(nex, ney, porder, ngp)