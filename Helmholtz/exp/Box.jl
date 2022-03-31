# using Helmholtz

include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

function Helmhotz_test(nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
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
    bc_funcs = [(x,y)-> 0.0,  (x,y)-> 0.0,  (x,y)-> 0.0,  (x,y)-> 0.0]
    ω = 1000.0
    c_func = (x,y)-> -sin(20π*x)sin(20π*y)
    x_0, y_0 = 0.3, 0.5
    s_func = (x,y)-> exp( -((x - x_0)^2 + (y - y_0)^2)/0.1^2 )

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
    visScalarField(domain, domain.state)

end

nex = ney = 100

Helmhotz_test(nex, ney, 2, 3)