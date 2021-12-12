include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")


function Helmhotz_D2N_Test(geometry::String, nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
    """
    test the D2N function output: g->∂u/∂n
    -Δu - ω^2/c^2 u = f  in Ω=[0,1]×[0,1]
    ω = 1
    c = 1 + x + y
    f = -sin(πx)sin(πy) - ω^2/(1 + x + y)^2 * (sin(πx)sin(πy)/(-2π^2) + x + y)
    
    u = sin(πx)sin(πy)/(-2π^2) + x + y
    
    on ∂Ω , u = x + y
    on ∂Ω , ∇u = (cos(πx)sin(πy)/(-2π) + 1, sin(πx)cos(πy)/(-2π))+1
    
    ===============For Box geometry
    ----- 3 -----
    |           |
    |           |
    4           2
    |           |
    |           |
    ----- 1 -----
    
    bottom N = (0, -1) :  ∂u/∂n = -sin(πx)/(-2π) - 1
    right  N = (1,  0) :  ∂u/∂n = -sin(πy)/(-2π) + 1
    top    N = (0,  1) :  ∂u/∂n = -sin(πx)/(-2π) + 1
    left   N = (-1, 0) :  ∂u/∂n = -sin(πy)/(-2π) - 1
    
    Bottom Right -> Dirichlet
    Top    Left  -> Neumann
    
    ===============For Ring geometry
    ****2****
    *           * 
    *    *  *  *    * 
    *    *  1  *    * 
    *    *  *  *   *
    *           *
    *********
    inner   N = -(x, y)/sqrt(x^2 + y^2)
    outter  N = (x, y)/sqrt(x^2 + y^2)
    
    inner -> Dirichlet
    outter -> Neumann
    """
    
    ω = 1.0
    c_func = (x,y)-> 1.0 + x^2 + y^2
    s_func = (x,y)-> -sin(π*x)sin(π*y) - ω^2/(1 + x^2 + y^2)^2 * (sin(π*x)sin(π*y)/(-2π^2) + x + y)
    uref_func = (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y
    ∇uref_func = (x,y)-> ( cos(π*x)sin(π*y)/(-2π) + 1,  sin(π*x)cos(π*y)/(-2π) + 1)
    
    
    if geometry == "box"
        bc_types = ["Dirichlet", "Dirichlet", "Neumann", "Neumann"]
        bc_funcs = [(x,y)-> x+y, (x,y)-> x+y, (x,y)-> -sin(π*x)/(-2π)+1, (x,y)-> -sin(π*y)/(-2π)-1 ]
        
        Lx, Ly = 1.0, 1.0
        nodes, elnodes, bc_nodes = box(Lx, Ly, nex, ney, porder)
        
        normal_ref_func = ((x,y) -> [0.0;-1.0],
        (x,y) -> [1.0; 0.0], 
        (x,y) -> [0.0; 1.0],
        (x,y) -> [-1.0; 0.0]
        )
    elseif geometry == "ring"
        
        bc_types = ["Dirichlet", "Neumann"]
        bc_funcs = [(x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y,
        (x,y)-> (cos(π*x)sin(π*y)/(-2π) + 1)*x/sqrt(x^2 + y^2) + (sin(π*x)cos(π*y)/(-2π) + 1)*y/sqrt(x^2 + y^2) ]
        
        
        r, R = 1.0, 2.0
        nodes, elnodes, bc_nodes = ring(r, R, nex, ney, porder)
        
        normal_ref_func = ((x,y) -> [-x/sqrt(x^2 + y^2) ; -y/sqrt(x^2 + y^2)],
        (x,y) -> [ x/sqrt(x^2 + y^2) ;  y/sqrt(x^2 + y^2)],
        )

    elseif geometry == "lshape"
        bc_types = ["Dirichlet", "Dirichlet", "Dirichlet", "Neumann", "Neumann", "Neumann"]
        bc_funcs = [(x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y,
                    (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y,
                    (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y,
                    (x,y)-> (cos(π*x)sin(π*y)/(-2π) + 1),
                    (x,y)-> ((sin(π*x)cos(π*y)/(-2π))+1),
                    (x,y)-> -(cos(π*x)sin(π*y)/(-2π) + 1)]
        
        Lx, Ly = 1.0, 1.0
        nodes, elnodes, bc_nodes = lshape(Lx, Ly, nex, ney, porder)
        normal_ref_func = ((x,y) -> [0 ; -1],
                           (x,y) -> [1 ;  0],
                           (x,y) -> [0 ;  1],
                           (x,y) -> [1 ;  0],
                           (x,y) -> [0 ;  1],
                           (x,y) -> [-1 ; 0]
        )

    else
        error("unrecognized geometry: ", geometry)
    end
    
    
    domain = Domain(nodes, elnodes,
    bc_nodes, bc_types, bc_funcs,
    porder, ngp; 
    ω = ω, 
    c_func = c_func, 
    s_func = s_func)
    
    domain = solve!(domain)
    
    
    DBC_ele = domain.DBC_ele
    elements = domain.elements
    
    nDBC_ele = size(DBC_ele, 1)
    
    DBC_∂u∂n_ele = computeNeumannOnDirichletEdge(domain)
    ∂u∂n_data = computeNeumannOnDirichletNode(domain, DBC_∂u∂n_ele, bc_nodes, bc_types)
    
    # compute reference values
    DBC_∂u∂n_ele_ref = zeros(Float64, nDBC_ele, porder+1)
    nbcs = length(bc_types)
    nnodes = domain.nnodes
    ∂u∂n_data_ref = zeros(Float64, nbcs, nnodes)
    
    
    for DBC_ele_id = 1:nDBC_ele
        
        elem_id, edge_id, bc_id = DBC_ele[DBC_ele_id, :]
        elem = domain.elements[elem_id]
        
        loc_node_ids = getLocalEdgeNodes(elem, edge_id)
        coords = elem.coords[loc_node_ids, :]
        for i = 1:porder+1
            ∇uref = ∇uref_func(coords[i, 1], coords[i, 2])
            normal_ref = normal_ref_func[bc_id](coords[i, 1], coords[i, 2])
            DBC_∂u∂n_ele_ref[DBC_ele_id, i] = ∇uref[1]*normal_ref[1] + ∇uref[2]*normal_ref[2]
        end
        
    end
    
    for bc_id = 1:nbcs
        
        for n_id = 1:nnodes
            if bc_nodes[n_id, bc_id]
                ∇uref = ∇uref_func(nodes[n_id, 1], nodes[n_id, 2])
                normal_ref = normal_ref_func[bc_id](nodes[n_id, 1], nodes[n_id, 2])
                ∂u∂n_data_ref[bc_id, n_id] = ∇uref[1]*normal_ref[1] + ∇uref[2]*normal_ref[2]
            end
        end
    end
    
    edge_error =  norm(DBC_∂u∂n_ele_ref -  DBC_∂u∂n_ele)/norm(DBC_∂u∂n_ele_ref)
    
    node_error =  norm(∂u∂n_data_ref[bc_types .== "Dirichlet", :] -  ∂u∂n_data[bc_types .== "Dirichlet", :])/norm(∂u∂n_data_ref[bc_types .== "Dirichlet", :])
    
    return edge_error, node_error
end


function Helmhotz_N2D_Test(geometry::String, nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3)
    """
    test the N2D function output: g->∂u/∂n
    -Δu - ω^2/c^2 u = f  in Ω=[0,1]×[0,1]
    ω = 1
    c = 1 + x + y
    f = -sin(πx)sin(πy) - ω^2/(1 + x + y)^2 * (sin(πx)sin(πy)/(-2π^2) + x + y)
    
    u = sin(πx)sin(πy)/(-2π^2) + x + y
    
    on ∂Ω , u = x + y
    on ∂Ω , ∇u = (cos(πx)sin(πy)/(-2π) + 1, sin(πx)cos(πy)/(-2π))+1
    
    ===============For Box geometry
    ----- 3 -----
    |           |
    |           |
    4           2
    |           |
    |           |
    ----- 1 -----
    
    bottom N = (0, -1) :  ∂u/∂n = -sin(πx)/(-2π) - 1
    right  N = (1,  0) :  ∂u/∂n = -sin(πy)/(-2π) + 1
    top    N = (0,  1) :  ∂u/∂n = -sin(πx)/(-2π) + 1
    left   N = (-1, 0) :  ∂u/∂n = -sin(πy)/(-2π) - 1
    
    Bottom Right -> Dirichlet
    Top    Left  -> Neumann
    
    ===============For Ring geometry
    ****2****
    *           * 
    *    *  *  *    * 
    *    *  1  *    * 
    *    *  *  *   *
    *           *
    *********
    inner   N = -(x, y)/sqrt(x^2 + y^2)
    outter  N = (x, y)/sqrt(x^2 + y^2)
    
    inner -> Dirichlet
    outter -> Neumann
    """
    
    ω = 1.0
    c_func = (x,y)-> 1.0 + x^2 + y^2
    s_func = (x,y)-> -sin(π*x)sin(π*y) - ω^2/(1 + x^2 + y^2)^2 * (sin(π*x)sin(π*y)/(-2π^2) + x + y)
    uref_func = (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y
    ∇uref_func = (x,y)-> ( cos(π*x)sin(π*y)/(-2π) + 1,  sin(π*x)cos(π*y)/(-2π) + 1)
    
    
    if geometry == "box"
        bc_types = ["Dirichlet", "Dirichlet", "Neumann", "Neumann"]
        bc_funcs = [(x,y)-> x+y, (x,y)-> x+y, (x,y)-> -sin(π*x)/(-2π)+1, (x,y)-> -sin(π*y)/(-2π)-1 ]
        
        Lx, Ly = 1.0, 1.0
        nodes, elnodes, bc_nodes = box(Lx, Ly, nex, ney, porder)
        
        normal_ref_func = ((x,y) -> [0.0;-1.0],
        (x,y) -> [1.0; 0.0], 
        (x,y) -> [0.0; 1.0],
        (x,y) -> [-1.0; 0.0]
        )
    elseif geometry == "ring"
        
        bc_types = ["Dirichlet", "Neumann"]
        bc_funcs = [(x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y,
        (x,y)-> (cos(π*x)sin(π*y)/(-2π) + 1)*x/sqrt(x^2 + y^2) + (sin(π*x)cos(π*y)/(-2π) + 1)*y/sqrt(x^2 + y^2) ]
        
        
        r, R = 1.0, 2.0
        nodes, elnodes, bc_nodes = ring(r, R, nex, ney, porder)
        
        normal_ref_func = ((x,y) -> [-x/sqrt(x^2 + y^2) ; -y/sqrt(x^2 + y^2)],
        (x,y) -> [ x/sqrt(x^2 + y^2) ;  y/sqrt(x^2 + y^2)],
        )

    elseif geometry == "lshape"
        bc_types = ["Dirichlet", "Dirichlet", "Dirichlet", "Neumann", "Neumann", "Neumann"]
        bc_funcs = [(x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y,
                    (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y,
                    (x,y)-> sin(π*x)sin(π*y)/(-2π^2) + x + y,
                    (x,y)-> (cos(π*x)sin(π*y)/(-2π) + 1),
                    (x,y)-> ((sin(π*x)cos(π*y)/(-2π))+1),
                    (x,y)-> -(cos(π*x)sin(π*y)/(-2π) + 1)]
        
        Lx, Ly = 1.0, 1.0
        nodes, elnodes, bc_nodes = lshape(Lx, Ly, nex, ney, porder)
        normal_ref_func = ((x,y) -> [0 ; -1],
                           (x,y) -> [1 ;  0],
                           (x,y) -> [0 ;  1],
                           (x,y) -> [1 ;  0],
                           (x,y) -> [0 ;  1],
                           (x,y) -> [-1 ; 0]
        )

    else
        error("unrecognized geometry: ", geometry)
    end
    
    
    domain = Domain(nodes, elnodes,
    bc_nodes, bc_types, bc_funcs,
    porder, ngp; 
    ω = ω, 
    c_func = c_func, 
    s_func = s_func)
    
    domain = solve!(domain)
    
    
    DBC_ele = domain.DBC_ele
    elements = domain.elements
    
    nDBC_ele = size(DBC_ele, 1)
    
    
    u_data = computeDirichletOnNeumannNode(domain, bc_nodes, bc_types)
    
    # compute reference values
    nbcs = length(bc_types)
    nnodes = domain.nnodes
    u_data_ref = zeros(Float64, nbcs, nnodes)
    
    
    # for DBC_ele_id = 1:nDBC_ele
    #     elem_id, edge_id, bc_id = DBC_ele[DBC_ele_id, :]
    #     elem = domain.elements[elem_id]
    #     loc_node_ids = getLocalEdgeNodes(elem, edge_id)
    #     coords = elem.coords[loc_node_ids, :]
    #     for i = 1:porder+1
    #         ∇uref = ∇uref_func(coords[i, 1], coords[i, 2])
    #         normal_ref = normal_ref_func[bc_id](coords[i, 1], coords[i, 2])
    #         DBC_∂u∂n_ele_ref[DBC_ele_id, i] = ∇uref[1]*normal_ref[1] + ∇uref[2]*normal_ref[2]
    #     end
    # end
    
    for bc_id = 1:nbcs
        for n_id = 1:nnodes
            if bc_nodes[n_id, bc_id]
                uref = uref_func(nodes[n_id, 1], nodes[n_id, 2])
                u_data_ref[bc_id, n_id] = uref
            end
        end
    end
    
    node_error =  norm(u_data_ref[bc_types .== "Neumann", :] -  u_data[bc_types .== "Neumann", :])/norm(u_data_ref[bc_types .== "Neumann", :])
    
    return node_error
end

function main()
    base_n = 10
    level_n = 3
    test_n = 2
    errors = zeros(Float64, test_n, level_n)
    ngp = 3
    for geometry in ("box","ring", "lshape")
        for porder = 1:2
            @info "Start N2D test ", " geometry = ", geometry, " porder = ", porder
            for level_id = 1:level_n
                nex, ney = base_n*2^level_id, 2*base_n*2^level_id
                errors[:, level_id] .= Helmhotz_D2N_Test(geometry, nex, ney, porder, ngp)
                
                @info "Helmhotz D2N test : ", "nex = ", nex, " ney = ", ney, " geometry = ", geometry, " porder = ", porder 
                @info "Edge test error is ", errors[1, level_id] , " Node test error is ", errors[2, level_id]
            end
            for level_id in 1:level_n-1
                edge_rate = log2(errors[1, level_id]) - log2(errors[1, level_id + 1])
                node_rate = log2(errors[2, level_id]) - log2(errors[2, level_id + 1])
                @info "Rates for level ", level_id, " are ", edge_rate, node_rate
            end
            
        end
    end

    errors = zeros(Float64, level_n)
    for geometry in ("box","ring", "lshape")
        for porder = 1:2
            @info "Start N2D test ", " geometry = ", geometry, " porder = ", porder
            for level_id = 1:level_n
                nex, ney = base_n*2^level_id, 2*base_n*2^level_id
                errors[level_id] = Helmhotz_N2D_Test(geometry, nex, ney, porder, ngp)
                
                @info "Helmhotz N2D test : ", "nex = ", nex, " ney = ", ney, " geometry = ", geometry, " porder = ", porder 
                @info " Node test error is ", errors[level_id]
            end
            for level_id in 1:level_n-1
                node_rate = log2(errors[level_id]) - log2(errors[level_id + 1])
                @info "Rates for level ", level_id, " are ", node_rate
            end
            
        end
    end
end

main()