using LinearAlgebra
using Random
using Distributions
using NPZ

function generate_KL_basis(N_x::IT, N_KL::IT, N_u::IT, N_a::IT; d::FT=2.0, τ::FT=3.0, seed::IT=123) where {FT<:AbstractFloat, IT<:Int}
    Random.seed!(seed);

    xx = LinRange(0, 1, N_x)
    uTs = LinRange(0, 1, N_u)
    φ0 = zeros(FT, N_KL, N_x)
    λ = zeros(FT, N_KL)
    for l = 1:div(N_KL,2)
        φ0[2l-1, :] = sqrt(2)*cos.(2 * pi * l * xx)
        φ0[2l, :]   = sqrt(2)*sin.(2 * pi * l * xx)
        λ[2l-1]     = (2^2*pi^2*l^2  + τ^2)^(-d)
        λ[2l]       = (2^2*pi^2*l^2  + τ^2)^(-d)
    end

    M = N_a * N_u

    θ0s = rand(Normal(0, 1), N_KL, M)
    a0s = zeros(Float64, N_x, M)
    aTs = zeros(Float64, N_x, M)

    φT = zeros(FT, N_KL, N_x)
    for i = 1:N_u   
        for l = 1:div(N_KL,2)
            φT[2l-1, :] = sqrt(2)*cos.(2 * pi * l * ((xx .- uTs[i] .+ 1).%1))
            φT[2l, :]   = sqrt(2)*sin.(2 * pi * l * ((xx .- uTs[i] .+ 1).%1))
        end

        for j = 1:N_a 
            k = j + (i - 1)*N_a
            θ0 = θ0s[:, k]
            for l = 1:N_KL
                a0s[:, k] .+= θ0[l]*sqrt(λ[l])*φ0[l, :]
                aTs[:, k] .+= θ0[l]*sqrt(λ[l])*φT[l, :]
            end
        end
    end

    return θ0s, repeat(uTs, outer=[1, N_a])'[:], a0s, aTs
end

N_x = 100
N_KL = 50 
N_u = 50 
N_a = 500
θ0s, uTs, a0s, aTs = generate_KL_basis(N_x, N_KL, N_u, N_a;)


npzwrite("../data/data.npz", Dict("theta" => θ0s, "u" => uTs, "a0" => a0s, "aT" => aTs))

