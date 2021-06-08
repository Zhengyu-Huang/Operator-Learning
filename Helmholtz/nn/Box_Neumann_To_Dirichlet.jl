# using Helmholtz
include("../src/Util.jl")
include("../src/ShapeFunctions.jl")
include("../src/Element.jl")
include("../src/Domain.jl")
include("../src/Solver.jl")
include("../src/Mesh.jl")

bump_func = (x,y, x0, d0) -> Float64.( abs.(x .- x0) .<= d0/2.0 )
function hat_func(x, y, x0, Δx)
    if x >= x0 && x <= x0 + Δx

	return 1.0 - (x - x0)/Δx

    elseif  x <= x0 && x >= x0 - Δx

        return 1.0 - (x0 - x)/Δx

    else

        return 0.0
    end
end

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









#=
Compute sorted pair (i, j), sorted by i^2 + j^2
wInt64h i≥0 and j≥0 and i+j>0

These pairs are used for Karhunen–Loève expansion
=#
function compute_seq_pairs(N_KL::Int64) 
    seq_pairs = zeros(Int64, N_KL, 2)
    trunc_Nx = trunc(Int64, sqrt(2*N_KL)) + 1
    
    include_00 = false
    seq_pairs = zeros(Int64, (trunc_Nx+1)^2 - 1 + include_00, 2)
    seq_pairs_mag = zeros(Int64, (trunc_Nx+1)^2 - 1 + include_00)
    
    seq_pairs_i = 0
    for i = 0:trunc_Nx
        for j = 0:trunc_Nx
            if (i == 0 && j ==0 && ~include_00)
                continue
            end
            seq_pairs_i += 1
            seq_pairs[seq_pairs_i, :] .= i, j
            seq_pairs_mag[seq_pairs_i] = i^2 + j^2
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    return seq_pairs[1:N_KL, :]
end



function c_func_uniform(θ::Float64) 
    
    c = 250 + θ 
    
    return c
end


#=
Generate parameters for logk field, based on Karhunen–Loève expansion.
They include eigenfunctions φ, eigenvalues λ and the reference parameters θ_ref, 
and reference field logk_2d field

logκ = ∑ u_l √λ_l φ_l(x)                l = (l₁,l₂) ∈ Z^{0+}×Z^{0+} \ (0,0)

where φ_{l}(x) = √2 cos(πl₁x₁)             l₂ = 0
                 √2 cos(πl₂x₂)             l₁ = 0
                 2  cos(πl₁x₁)cos(πl₂x₂) 
      λ_{l} = (π^2l^2 + τ^2)^{-d} 

They can be sorted, where the eigenvalues λ_{l} are in descending order

generate_θ_KL function generates the summation of the first N_KL terms 
=#
function c_func_random(x1::Float64, x2::Float64, θ::Array{Float64, 1}, seq_pairs::Array{Int64, 2}, d::Float64=2.0, τ::Float64=3.0) 
    
    N_KL = length(θ)
    
    a = 0
    
    for i = 1:N_KL
        λ = (pi^2*(seq_pairs[i, 1]^2 + seq_pairs[i, 2]^2) + τ^2)^(-d)
        
        if (seq_pairs[i, 1] == 0 && seq_pairs[i, 2] == 0)
            a += θ[i] * sqrt(λ)
        elseif (seq_pairs[i, 1] == 0)
            a += θ[i] * sqrt(λ) * sqrt(2)*cos.(pi * (seq_pairs[i, 2]*x2))
        elseif (seq_pairs[i, 2] == 0)
            a += θ[i] * sqrt(λ) * sqrt(2)*cos.(pi * (seq_pairs[i, 1]*x1))
        else
            a += θ[i] * sqrt(λ) * 2*cos.(pi * (seq_pairs[i, 1]*x1)) .*  cos.(pi * (seq_pairs[i, 2]*x2))
        end

        
    end
    
    c = 250 .+ 50*(a > 0.0)
    
    return c
end


function compute_a_field(xx::Array{Float64,1}, θ::Array{Float64, 1}, seq_pairs::Array{Int64, 2}, d::Float64=2.0, τ::Float64=3.0)
    N = length(xx)
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    
    N_KL = length(θ)
    φ = zeros(Float64, N_KL, N, N)
    λ = zeros(Float64, N_KL)
    
    
    for i = 1:N_KL
        if (seq_pairs[i, 1] == 0 && seq_pairs[i, 2] == 0)
            φ[i, :, :] .= 1.0
        elseif (seq_pairs[i, 1] == 0)
            φ[i, :, :] = sqrt(2)*cos.(pi * (seq_pairs[i, 2]*Y))
        elseif (seq_pairs[i, 2] == 0)
            φ[i, :, :] = sqrt(2)*cos.(pi * (seq_pairs[i, 1]*X))
        else
            φ[i, :, :] = 2*cos.(pi * (seq_pairs[i, 1]*X)) .*  cos.(pi * (seq_pairs[i, 2]*Y))
        end

        λ[i] = (pi^2*(seq_pairs[i, 1]^2 + seq_pairs[i, 2]^2) + τ^2)^(-d)
    end
    

    loga_2d = zeros(Float64, N, N)
    for i = 1:N_KL
        loga_2d .+= θ[i]*sqrt(λ[i])*φ[i, :, :]
    end
    
    return loga_2d
    
end



function Data_Generate(generate_method::String, N_data::Int64, N_θ::Int64;
    ne::Int64 = 100,   seed::Int64=123)
    @assert(generate_method == "Uniform" || generate_method == "Random")
    
    porder = 1
    Δx = 1.0/ne
    K_scale = zeros(Float64, ne*porder+1) .+ Δx
    K_scale[1] = K_scale[end] = Δx/2.0
    Random.seed!(seed)

    if generate_method == "Uniform"
        assert(N_θ == 1)
        θ = rand(Uniform(0, 50), N_data, N_θ);
        κ = zeros(ne+1, ne+1, N_data)
        for i = 1:N_data
            cs = [(x,y)->c_func_uniform(θ[i]);]
            # generate Dirichlet to Neumman results output for different condInt64ions
            # data =[nodal posInt64ions, (x, ∂u∂n, u), 4 edges, experiment number]
            data = Generate_Input_Output(cs, ne, porder);
            
            # data =[nodal posInt64ions, (x, ∂u∂n, u), 4 edges, experiment number]
            bc_id = 3
            u_n = data[:, 2, bc_id, :]
            u_d = data[:, 3, bc_id, :]
            K = u_d/u_n
            κ[:, :, i] = K ./ K_scale' 
        end 
        


    elseif generate_method == "Random"
        θ = rand(Normal(0, 1.0), N_data, N_θ);
        κ = zeros(ne+1, ne+1, N_data)

        seq_pairs = compute_seq_pairs(N_θ)
        for i = 1:N_data
            @info "i = ", i
            cs = [(x,y)->c_func_random(x, y, θ[i, :], seq_pairs);]

            # generate Dirichlet to Neumman results output for different condInt64ions
            # data =[nodal posInt64ions, (x, ∂u∂n, u), 4 edges, experiment number]
            data = Generate_Input_Output(cs, ne, porder);
            
            # data =[nodal posInt64ions, (x, ∂u∂n, u), 4 edges, experiment number]
            bc_id = 3
            u_n = data[:, 2, bc_id, :]
            u_d = data[:, 3, bc_id, :]
            K = u_d/u_n
            κ[:, :, i] = K ./ K_scale' 
        end 

    else 
        @info "generate_method: $(generate_method) and data_type == $(data_type) have not implemented yet"
    end
    
    return θ, κ
    
    
end
