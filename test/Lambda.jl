include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

"""
Test the Lambda function output
    g -> ∂u/∂n
"""

function Helmhotz_Lambda_test()
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
    
    
    
    
    """
    ----- 3 -----
    |           |
    |           |
    4           2
    |           |
    |           |
    ----- 1 -----
    """
    
    
    
    bc_types = ["Dirichlet", "Dirichlet", "Neumann", "Neumann"]
    bc_funcs = [(x,y)-> x+y, (x,y)-> x+y, (x,y)-> -sin(π*x)/(-2π)+1, (x,y)-> -sin(π*y)/(-2π)-1 ]
    ω = 1.0
    c_func = (x,y)-> 1.0 + x + y
    s_func = (x,y)-> -sin(π*x)sin(π*y) - ω^2/(1 + x + y)^2 * (sin(π*x)sin(π*y)/(-2π^2) + x + y)
    uref_func = (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y
    
    ∇uref_func = (x,y)-> ( cos(π*x)sin(π*y)/(-2π) + 1,  sin(π*x)cos(π*y)/(-2π) + 1)
    
    nex, ney = 40, 40
    Lx, Ly = 1.0, 1.0
    porder = 2
    nodes, elnodes, bc_nodes = box(Lx, Ly, nex, ney, porder)
    ngp = 3
    
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
    
    #bottom N = (0, -1) :  ∂u/∂n = -sin(πx)/(-2π) - 1
    
    #right  N = (1,  0) :  ∂u/∂n = -sin(πy)/(-2π) + 1
    
    # @info "error is ", norm(state_ref - domain.state)
    
    DBC_ele = domain.DBC_ele
    elements = domain.elements

    nDBC_ele = size(DBC_ele, 1)
    DBC_∂u∂n_ele = zeros(Float64, nDBC_ele, porder+1)
    DBC_∂u∂n_ele_ref = zeros(Float64, nDBC_ele, porder+1)
    normal_ref = [0.0 -1.0; 1.0 0.0; 0.0 1.0; -1.0 0.0]
    
    for DBC_ele_id = 1:nDBC_ele
    
        elem_id, edge_id, bc_id = DBC_ele[DBC_ele_id, :]
        elem = domain.elements[elem_id]
        u = domain.state[elem.elnodes]
        DBC_∂u∂n_ele[DBC_ele_id, :] = compute∂u∂n(elements[elem_id], edge_id, u)
        
        loc_node_ids = getLocalEdgeNodes(elem, edge_id)
        coords = elem.coords[loc_node_ids, :]
        for i = 1:porder+1
            ∇uref = ∇uref_func(coords[i, 1], coords[i, 2])
            DBC_∂u∂n_ele_ref[DBC_ele_id, i] = ∇uref[1]*normal_ref[edge_id, 1] + ∇uref[2]*normal_ref[edge_id, 2]
        end

    end

    @info "rel. error is ", norm(DBC_∂u∂n_ele_ref -  DBC_∂u∂n_ele)/norm(DBC_∂u∂n_ele_ref)
end

Helmhotz_Lambda_test()
