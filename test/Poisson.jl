include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")


function Poisson_test(nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
    """
    -Δu - ω^2/c^2 u = f  in Ω=[0,1]×[0,1]
    
    ω = 0
    f = -sin(πx)sin(πy) 
    
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
    
    
    
    
    """
    bc_types--- 3 -----
    |           |
    |           |
    4           2
    |           |
    |           |
    ----- 1 -----
    """
    bc_types = ["Dirichlet", "Dirichlet", "Neumann", "Neumann"]
    bc_funcs = [(x,y)-> x+y, (x,y)-> x+y, (x,y)-> -sin(π*x)/(-2π)+1, (x,y)-> -sin(π*y)/(-2π)-1 ]
    ω = 0.0
    c_func = (x,y)-> 1.0
    s_func = (x,y)-> -sin(π*x)sin(π*y)
    uref_func = (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y
    
    Lx, Ly = 1.0, 1.0
    
    nodes, elnodes, bc_nodes = box(Lx, Ly, nex, ney, porder)
    
    domain = Domain(nodes, elnodes,
    bc_nodes, bc_types, bc_funcs,
    porder, ngp; 
    ω = ω, 
    c_func = c_func, 
    s_func = s_func)
    
    domain = solve!(domain)
    
    
    state_ref = copy(domain.state)
    for i = 1:domain.nnodes
        state_ref[i] = uref_func(domain.nodes[i, :]...)
    end
    
    error =  norm(state_ref - domain.state)/norm(state_ref)
    
end

function main()
    base_n = 10
    level_n = 3
    errors = zeros(Float64, level_n)
    ngp = 3
    for porder = 1:2
        for level_id = 1:level_n
            nex, ney = base_n*2^level_id, base_n*2^level_id
            errors[level_id] = Poisson_test(nex, ney, porder, ngp)

            @info "Poisson test : ", "nex = ", nex, " ney = ", ney, " porder = ", porder 
            @info "Error is ", errors[level_id]
        end
        for level_id in 1:level_n-1
            rate = log2(errors[level_id]) - log2(errors[level_id + 1])
            @info "rates for level ", level_id, " is ", rate
        end
    end
end

main()