using Random, Distributions, NPZ
include("Box_Neumann_To_Dirichlet.jl")


Ndata = 4000
N_θ = 100
generate_method = "Random"
ne=100
θ, κ = Data_Generate(generate_method, Ndata, N_θ; ne=ne)

npzwrite("$(generate_method)_Helmholtz_theta_$(N_θ).npy", θ)
npzwrite("$(generate_method)_Helmholtz_K_$(N_θ).npy", κ)
