# using Helmholtz

include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

function D2N(xi::Array{Float64, 1}, c::Float64, nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3;
    visualize::Bool = false )
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
    # ∑_l=1  ξ_i /(lπ)^2 sin(l*π*x) 
    fl = (x,l)-> (l*π)^-2 * xi[l] * sin.(l*π*x)
    f = x-> sum(l->fl(x,l),1:length(xi))

    bc_funcs = [(x,y)->0, (x,y)->0, (x,y)->f(x), (x,y)->0]
    ω = 1000.0
    c_func = (x,y)-> c
    s_func = (x,y)-> 0
    
    Lx, Ly = 1.0, 1.0
    nodes, elnodes, bc_nodes = box(Lx, Ly, nex, ney, porder)
    
    domain = Domain(nodes, elnodes,
    bc_nodes, bc_types, bc_funcs,
    porder, ngp; 
    ω = ω, 
    c_func = c_func, 
    s_func = s_func)
    
    domain = solve!(domain)

    if visualize
        PyPlot.figure()
        PyPlot.title("c = "*string(c))
        visScalarField(domain, domain.state, savefile="sinBox"*string(c)*".png")
        
    end
    
    DBC_∂u∂n_ele = computeNeumannOnDirichletEdge(domain)
    ∂u∂n_data = computeNeumannOnDirichletNode(domain, DBC_∂u∂n_ele, bc_nodes, bc_types)
    
    output_data = zeros(nex*porder+1, 2, 4) # 4 edges, nodal positions, (x, ∂u∂n) 
    for bc_id = 1:4
        bc_id_nodes = bc_nodes[:, bc_id]
        xx, yy = nodes[bc_id_nodes, 1], nodes[bc_id_nodes, 2]
        data_Dirichlet = [bc_funcs[bc_id](xx[i], yy[i]) for i =1:length(xx)]
        data_Neumann = ∂u∂n_data[bc_id, bc_id_nodes]

   
        output_data[:, 1, bc_id] .= (bc_id == 1 || bc_id == 3) ? xx : yy
        output_data[:, 2, bc_id] .= data_Neumann
        # sort the data
        output_data[:, :, bc_id] .= output_data[sortperm(output_data[:, 1, bc_id]), :, bc_id]
    end 

    return output_data
end


function Generate_Output(cs::Array{Float64,1}, N_l::Int64, ne::Int64, porder::Int64)
    # domain discretization set-up
    Lx, Ly = 1.0, 1.0   # box edge lengths
    nex, ney = ne, ne   # grid size
    
    
    # number of Dirichlet boundary edges
    n_dbcs = 4
    n_points = nex*porder + 1
    output_data_all = zeros(Float64, n_points, length(cs)* N_l,   n_dbcs)
    # generate data

    
    for ind_c = 1:length(cs)
        c = cs[ind_c]
        
        for l = 1:N_l
            
            xi = zeros(Float64, N_l)
            xi[l] = 1.0
            
            output_data = D2N(xi, c, nex, ney, porder; visualize = (l == 1))
            for i = 1:n_dbcs
                
                output_data_all[:, (ind_c - 1)*N_l + l, i] = output_data[:, 2, i]
            end
        end
    end

    return output_data_all
end


