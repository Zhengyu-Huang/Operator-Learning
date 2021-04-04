# using Helmholtz
include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

bump_func = (x,y, x0, d0) -> Float64.( abs.(x .- x0) .<= d0/2.0 )


function N2D(x0::Float64, d0::Float64, c_func::Function, nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3;
    visualize::Bool = false )
    """
    -Δu - ω^2/c^2 u = 0  in Ω=[0,1]×[0,1]
    
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
    bc_types = ["Neumann", "Neumann", "Neumann", "Neumann"]
    
    # define fourier boundary
    # f(x) = 1_{ |x - x0| < d0/2.0 } 
    bc_funcs = [(x,y)->0, (x,y)->0, (x,y)->bump_func(x, y, x0,d0), (x,y)->0]

    ω = 1000.0
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
        PyPlot.title("x0 = "*string(x0)*"d0 ="*string(d0))
        visScalarField(domain, domain.state, savefile="x0 = "*string(x0)*"d0 ="*string(d0)*".png")
        
    end
    
    u_data = computeDirichletOnNeumannNode(domain, bc_nodes, bc_types)
    
    data = zeros(nex*porder+1, 3, 4) # nodal positions, (x, ∂u∂n, u), 4 edges
    for bc_id = 1:4
        bc_id_nodes = bc_nodes[:, bc_id]
        xx, yy = nodes[bc_id_nodes, 1], nodes[bc_id_nodes, 2]
        data_Neumann = [bc_funcs[bc_id](xx[i], yy[i]) for i =1:length(xx)]
        data_Dirichlet = u_data[bc_id, bc_id_nodes]

   
        data[:, 1, bc_id] .= (bc_id == 1 || bc_id == 3) ? xx : yy
        if bc_id == 3
            data[:, 2, bc_id] .= bump_func(data[:, 1, bc_id], ones(Float64, size(data[:, 1, bc_id])), x0, d0)
        end
        data[:, 3, bc_id] .= data_Dirichlet
        # sort the data
        data[:, :, bc_id] .= data[sortperm(data[:, 1, bc_id]), :, bc_id]
    end 

    return data
end


function Generate_Input_Output(func_cs, x0d0s::Array{Float64, 2}, ne::Int64, porder::Int64)
    # domain discretization set-up
    Lx, Ly = 1.0, 1.0   # box edge lengths
    nex, ney = ne, ne   # grid size
    
    
    # number of Dirichlet boundary edges
    n_dbcs = 4
    n_points = nex*porder + 1
    data_all = zeros(Float64, n_points,  3, n_dbcs, length(func_cs)*size(x0d0s, 1)) # nodal positions, (x, ∂u∂n, u), 4 edges, experiment number
    # generate data

    
    for ind_c = 1:length(func_cs)
        func_c = func_cs[ind_c]
        
        for l = 1:size(x0d0s, 1)
            x0, d0 = x0d0s[l, :]
            
            data = N2D(x0, d0, func_c, nex, ney, porder; visualize = false)
            for i = 1:n_dbcs
                
                data_all[:,  :,  :, (ind_c - 1)*N_l + l] .= data
            end
        end
    end

    return data_all
end




###################################################################################

function hat_func(x, y, x0, Δx) 
    if x >= x0 && x <= x0 + Δx
        
        return 1.0 - (x - x0)/Δx

    elseif  x <= x0 && x >= x0 - Δx

        return 1.0 - (x0 - x)/Δx

    else 
    
        return 0.0
    end
end


# Generate with hat function at the ih-th point, ih = 0, 1 ... nex
function N2D(ih::Int64, c_func::Function, nex::Int64 = 20, ney::Int64 = 20, porder::Int64 = 2, ngp::Int64 = 3;
    visualize::Bool = false )
    """
    -Δu - ω^2/c^2 u = 0  in Ω=[0,1]×[0,1]
    
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
    ω = 1000.0
    s_func = (x,y)-> 0
    
    Lx, Ly = 1.0, 1.0

    Δx = Lx / nex
 
    bc_types = ["Neumann", "Neumann", "Neumann", "Neumann"]
    
    # define fourier boundary
    # f(x) = 1_{ |x - x0| < d0/2.0 } 
    bc_funcs = [(x,y)->0, (x,y)->0, (x,y)->hat_func(x, y, Δx*ih, Δx), (x,y)->0]


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
        PyPlot.title("hat func $(ih)")
        visScalarField(domain, domain.state, savefile="hat_func_$(ih).png")
        
    end
    
    u_data = computeDirichletOnNeumannNode(domain, bc_nodes, bc_types)
    
    data = zeros(nex*porder+1, 3, 4) # nodal positions, (x, ∂u∂n, u), 4 edges
    for bc_id = 1:4
        bc_id_nodes = bc_nodes[:, bc_id]
        xx, yy = nodes[bc_id_nodes, 1], nodes[bc_id_nodes, 2]
        data_Neumann = [bc_funcs[bc_id](xx[i], yy[i]) for i =1:length(xx)]
        data_Dirichlet = u_data[bc_id, bc_id_nodes]

   
        data[:, 1, bc_id] .= (bc_id == 1 || bc_id == 3) ? xx : yy
        if bc_id == 3
            for ip = 1: size(data, 1)
                data[ip, 2, bc_id] = hat_func(data[ip, 1, bc_id], 0.0, Δx*ih, Δx)
            end
        end
        data[:, 3, bc_id] .= data_Dirichlet
        # sort the data
        data[:, :, bc_id] .= data[sortperm(data[:, 1, bc_id]), :, bc_id]
    end 

    return data
end


function Generate_Input_Output(func_cs, ne::Int64, porder::Int64)

    @assert(porder == 1)
    # domain discretization set-up
    Lx, Ly = 1.0, 1.0   # box edge lengths
    nex, ney = ne, ne   # grid size
    
    # number of Dirichlet boundary edges
    n_dbcs = 4
    n_points = nex*porder + 1
    data_all = zeros(Float64, n_points,  3, n_dbcs, length(func_cs)*n_points) # nodal positions, (x, ∂u∂n, u), 4 edges, experiment number
    # generate data

    
    for ind_c = 1:length(func_cs)
        func_c = func_cs[ind_c]
        
        for l = 1:n_points
            # data = N2D(l - 1, func_c, nex, ney, porder; visualize = (l == 1))
            data = N2D(l - 1, func_c, nex, ney, porder; visualize = false)
            for i = 1:n_dbcs
                
                data_all[:,  :,  :, (ind_c - 1)*n_points + l] .= data
            end
        end
    end

    return data_all
end

