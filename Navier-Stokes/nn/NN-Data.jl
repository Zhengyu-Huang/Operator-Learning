using LinearAlgebra
using Distributions
using Random
using SparseArrays
using NPZ
include("../src/Spectral-Navier-Stokes.jl");


ν = 1.0/40                                      # viscosity
N, L = 128, 2*pi                                 # resolution and domain size 
ub, vb = 0.0, 0.0                               # background velocity 
method="Crank-Nicolson"                         # RK4 or Crank-Nicolson
N_t = 5000;                                     # time step
T = 10.0;                                        # final time
obs_ΔNx, obs_ΔNy, obs_ΔNt = 16, 16, 5000        #observation

plot_data = false

# The forcing has N_θ terms
N_θ = 100
seq_pairs = Compute_Seq_Pairs(Int64(N_θ/2))

Random.seed!(42);
N_data = 4
θθ = rand(TruncatedNormal(0,1, -1, 1), N_data, N_θ)
mesh = Spectral_Mesh(N, N, L, L)
ωω = zeros(N,N, N_data)

for i_d = 1:N_data
    @info "data ", i_d, " / ", N_data
    θ = θθ[i_d, :]
    
    # this is used for generating random initial condition
    s_param = Setup_Param(ν, ub, vb,  
        N, L,  
        method, N_t,
        obs_ΔNx, obs_ΔNy, obs_ΔNt, 
        0,
        100;)


    ω0_ref = s_param.ω0_ref

    solver = Spectral_NS_Solver(mesh, ν; fx = s_param.fx, fy = s_param.fy, ω0 = ω0_ref, ub = ub, vb = vb)  
    # The forcing fx and fy are set to be zero, they are only used for pressure computation
    # We specify the curl_f_hat directely (do not compute pressure)
    curl_f = Initial_ω0_KL(mesh, θ, seq_pairs)
    Trans_Grid_To_Spectral!(mesh, curl_f,  solver.curl_f_hat)
     
    
    if plot_data && i_d == 1
        PyPlot.figure(figsize = (4,3))
        Visual(mesh, solver.ω, "ω")
        PyPlot.title("Initial ω")
    end
    


    Δt = T/N_t 
    for i = 1:N_t
        Solve!(solver, Δt, method)
        if plot_data && i == N_t
            PyPlot.figure(figsize = (4,3))
            Visual(mesh, curl_f, "curl_f")
            PyPlot.title("∇×f")
            
            Update_Grid_Vars!(solver)
            PyPlot.figure(figsize = (4,3))
            Visual(mesh, solver.ω, "ω")
            PyPlot.title("ω")
        end
    end
    ωω[:, : , i_d] .= solver.ω
end





npzwrite("NS_theta_$(N_θ).npy", θθ)
npzwrite("NS_omega_$(N_θ).npy", ωω)

