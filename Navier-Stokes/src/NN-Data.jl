using LinearAlgebra
using Distributions
using Random
using SparseArrays
using NPZ

include("./Navier-Stokes-Force-Sol.jl");

function Data_Generate()
    N_data = 10
    ν = 1.0/40                                      # viscosity
    N, L = 64, 2*pi                                 # resolution and domain size 
    N_t = 5000;                                     # time step
    T = 10.0;                                        # final time

    d=2.0
    τ=3.0
    # The forcing has N_θ terms
    N_θ = 100
    seq_pairs = Compute_Seq_Pairs(100)

    Random.seed!(42);
    θf = rand(Normal(0,1), N_data, N_θ)
    curl_f = zeros(N, N, N_data)
    for i = 1:N_data
        curl_f[:,:, i] .= generate_ω0(L, N, θf[i,:], seq_pairs, d, τ)
    end

    θω = rand(Normal(0,1), N_θ)
    ω0 = generate_ω0(L, N, θω, seq_pairs, d, τ)

    # Define caller function
    g_(x::Matrix{FT}) where FT<:Real = 
        NS_Solver(x, ω0;  ν = ν, N_t = N_t, T = T)

    curl_f = curl_f

    ω_tuple = []
    for i = 1:N_data
        push!(ω_tuple, g_(curl_f[:,:,i])) # Outer dim is params iterator
    end

    ω_field = zeros(N, N, N_data)
    for i = 1:N_data
        ω_field[:,:, i] = ω_tuple[i]
    end

    npzwrite("Random_NS_theta_$(N_θ).npy",  θf)
    npzwrite("Random_NS_omega_$(N_θ).npy",  ω_field)
    npzwrite("Random_NS_curl_f_$(N_θ).npy", curl_f)

end

Data_Generate()