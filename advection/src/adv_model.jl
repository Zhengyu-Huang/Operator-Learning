using LinearAlgebra
using Random
using Distributions
using NPZ

function generate_data(N_x::IT, N_KL::IT, M::IT; u::FT = 0.5, d::FT=1.0, τ::FT=3.0, seed::IT=123) where {FT<:AbstractFloat, IT<:Int}
    Random.seed!(seed);

    xx = LinRange(0, 1, N_x)
    φ0 = zeros(FT, N_KL, N_x)
    φT = zeros(FT, N_KL, N_x)
    λ = zeros(FT, N_KL)
    for l = 1:div(N_KL,2)
        φ0[2l-1, :] = sqrt(2)*cos.(2 * pi * l * xx)
        φ0[2l, :]   = sqrt(2)*sin.(2 * pi * l * xx)
        λ[2l-1]     = (2^2*pi^2*l^2  + τ^2)^(-d)
        λ[2l]       = (2^2*pi^2*l^2  + τ^2)^(-d)
        

        φT[2l-1, :] = sqrt(2)*cos.(2 * pi * l * ((xx .- u .+ 1).%1))
        φT[2l, :]   = sqrt(2)*sin.(2 * pi * l * ((xx .- u .+ 1).%1))

    end

    θ0s = rand(Normal(0, 1), N_KL, M)
    a0s = zeros(Float64, N_x, M)
    aTs = zeros(Float64, N_x, M)

    

    for j = 1:M 
        θ0 = θ0s[:, j]
        for l = 1:N_KL
            a0s[:, j] .+= θ0[l]*sqrt(λ[l])*φ0[l, :]
            aTs[:, j] .+= θ0[l]*sqrt(λ[l])*φT[l, :]
        end
    end

    p_ind         = a0s .> 0
    a0s[p_ind]   .=  1.0
    a0s[.~p_ind] .= -1.0
    
    p_ind         = aTs .> 0
    aTs[p_ind]   .=  1.0
    aTs[.~p_ind] .= -1.0

    return θ0s, a0s, aTs
end

N_x  = 200
N_KL = 100 
M    = 40000
θ0s, a0s, aTs = generate_data(N_x, N_KL, M; u=0.5, d=2.0, τ=3.0)


npzwrite("adv_a0.npy", a0s)
npzwrite("adv_aT.npy", aTs)
