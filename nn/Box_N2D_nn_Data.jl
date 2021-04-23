using Random, Distributions, NPZ
include("Box_Neumann_To_Dirichlet.jl")


function c_func_uniform(θ::Float64) 
    
    c = 250 + θ 
    
    return c
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
function c_func_random(x1::Float64, x2::Float64, θ::Array{Float64, 1}, seq_pairs::Array{Int64, 2}, c_min::Float64, Δc::Float64, d::Float64=2.0, τ::Float64=3.0) 
    
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
       
    # c = 275 .+ 25*a
    
    c = c_min .+ Δc*(a > 0.0)

    
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



function Data_Generate(generate_method::String, data_type::String, N_data::Int64, N_θ::Int64; prefix::String="", ne::Int64 = 100,   seed::Int64=123)
    @assert(generate_method == "Uniform" || generate_method == "Random")
    @assert(data_type == "Direct" || generate_method == "Indirect")
    
    porder = 1
    Δx = 1.0/ne
    K_scale = zeros(Float64, ne*porder+1) .+ Δx
    K_scale[1] = K_scale[end] = Δx/2.0
    Random.seed!(seed)

    if generate_method == "Uniform" && data_type == "Direct"
        N_θ = 1
        # θ = rand(Uniform(0, 50), N_data, N_θ);
        θ = Array(LinRange(0, 50, N_data));
        κ = zeros(ne+1, ne+1, N_data)
        for i = 1:N_data
            @info "i = ", i
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
        
        npzwrite(prefix*"uniform_direct_theta.npy", θ)
        npzwrite(prefix*"uniform_direct_K.npy", κ)

    elseif generate_method == "Random" && data_type == "Direct"
        N_θ = 8
        θ = rand(Normal(0, 1.0), N_data, N_θ);
        κ = zeros(ne+1, ne+1, N_data)
	
	    cmin, Δc = 250.0, 50.0
        seq_pairs = compute_seq_pairs(N_θ)
        Threads.@threads for i = 1:N_data
            @info "i = ", i
            cs = [(x,y)->c_func_random(x, y, θ[i, :], seq_pairs, cmin, Δc)]

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
        
        npzwrite(prefix*"random_direct_theta.npy", θ)
        npzwrite(prefix*"random_direct_K.npy", κ)

    else 
        @info "generate_method: $(generate_method) and data_type == $(data_type) have not implemented yet"
    end
    
    return θ, κ
    
    
end





#Data_Generate("Random", "Direct", 100, 0; ne = 100,   seed = 16)
#Data_Generate("Random", "Direct", 100, 0; ne = 100,   seed = 61)
#Data_Generate("Random", "Direct", 100, 0; ne = 100,   seed = 31)
Data_Generate("Random", "Direct", 1000, 0; ne = 100)

# Data_Generate("Uniform", "Direct", 501, 0; ne = 100,   seed = 123)

# Data_Generate("Uniform", "Direct", 10, 0; prefix = "test_", ne = 100,   seed = 42)

# Data_Generate("Random", "Direct", 10, 0; prefix = "test_", ne = 100,   seed = 42)
