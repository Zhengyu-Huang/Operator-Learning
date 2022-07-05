using LinearAlgebra
using Distributions
using Random
using SparseArrays
using NPZ
include("Spectral-Navier-Stokes.jl");


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
function generate_ω0(L::Float64, N::Int64, θ::Array{Float64, 1}, 
                       seq_pairs::Array{Int64, 2}, d::Float64=2.0, τ::Float64=3.0) 
    
    N_x, N_y = N, N
    ω0 = zeros(Float64, N_x, N_y)
    X, Y = zeros(Float64, N_x, N_y), zeros(Float64, N_x, N_y)
    Δx, Δy = L/N_x, L/N_y
    for ix = 1:N_x
        for iy = 1:N_y
            X[ix, iy] = (ix-1)*Δx
            Y[ix, iy] = (iy-1)*Δy
        end
    end

    N_θ = length(θ)
    @assert(N_θ % 2 == 0)
    N_KL = div(N_θ, 2)
    abk = rand(Normal(0, 1), N_KL, 2)

    for i = 1:N_KL
        kx, ky = seq_pairs[i,:]
        
        @assert(kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
        
        ak, bk = abk[i, 1], abk[i, 2]
        ω0 .+= 1/(sqrt(2)*pi) * (ak * cos.(kx*X + ky*Y) + bk * sin.(kx*X + ky*Y))/(τ^2 + (kx^2 + ky^2))^(d/2)
    end
    
    return ω0
end


function NS_Solver(curl_f, ω0;  ν = 1.0/40, N_t = 5000, T = 10.0)
    L = 2*pi                                        # viscosity
    N = size(ω0, 1)                                 # resolution and domain size 
    ub, vb = 0.0, 0.0                               # background velocity 
    method="Crank-Nicolson"                         # RK4 or Crank-Nicolson

    mesh = Spectral_Mesh(N, N, L, L)

    solver = Spectral_NS_Solver(mesh, ν; curl_f = curl_f, ω0 = ω0, ub = ub, vb = vb)  

    Δt = T/N_t 
    for i = 1:N_t
        Solve!(solver, Δt, method)
    end

    Update_Grid_Vars!(solver)
    return solver.ω
end
