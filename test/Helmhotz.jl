include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")



function Helmhotz_test()
    """
    -Δu - ω^2/c^2 u = f  in Ω=[0,1]×[0,1]
    
    ω = 1
    c = 1 + x + y
    f = -sin(πx)sin(πy) - ω^2/(1 + x + y)^2 * (sin(πx)sin(πy)/(-2π^2) + x + y)
    
    u = sin(πx)sin(πy)/(-2π^2) + x + y
    
    on ∂Ω , u = x + y
    on ∂Ω , ∇u = (cos(πx)sin(πy)/(-2π) + 1, sin(πx)cos(πy)/(-2π))+1
            bottom N = (0, -1) :  ∂u/∂n = -sin(πx)/(-2π) - 1
            right  N = (1,  0) :  ∂u/∂n = -sin(πy)/(-2π) + 1
            top    N = (0,  1) :  ∂u/∂n = -sin(πx)/(-2π) + 1
            left   N = (-1, 0) :  ∂u/∂n = -sin(πy)/(-2π) - 1
    
    Bottom Right -> Dirichlet
    Top    Left  -> Neumann
    """
    
    # using Helmholtz
    
    
    
    nex, ney = 20, 20
    """
    ----- 3 -----
    |           |
    |           |
    4           2
    |           |
    |           |
    ----- 1 -----
    """
    bc = ["Dirichlet", "Dirichlet", "Neumann", "Neumann"]
    bc_func = [(x,y)-> x+y, (x,y)-> x+y, (x,y)-> -sin(π*x)/(-2π)+1, (x,y)-> -sin(π*y)/(-2π)-1 ]
    ω = 1.0
    c_func = (x,y)-> 1.0 + x + y
    s_func = (x,y)-> -sin(π*x)sin(π*y) - ω^2/(1 + x + y)^2 * (sin(π*x)sin(π*y)/(-2π^2) + x + y)
    uref_func = (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y
    
    nex, ney = 40, 40
    nodes, elements, DBC, DBC_ele, u_g,  NBC, NBC_ele, ∂u∂n_ele = box(nex, ney, bc, bc_func, ω, c_func, s_func; porder = 2, ngp = 3, Lx = 1.0, Ly = 1.0)
    domain = Domain(nodes,  elements,  DBC,   DBC_ele, u_g,  NBC,  NBC_ele,  ∂u∂n_ele, s_func)
    domain = solve!(domain)
    
    
    state_ref = copy(domain.state)
    for i = 1:domain.nnodes
        state_ref[i] = uref_func(domain.nodes[i, :]...)
    end
    
    @info "error is ", norm(state_ref - domain.state)
    
    visScalarField(domain, domain.state - state_ref)
    
    end

    Helmhotz_test()
