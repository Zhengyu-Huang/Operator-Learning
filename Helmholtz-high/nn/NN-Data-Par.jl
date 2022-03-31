@everywhere using Random, Distributions, NPZ
@everywhere include("Box_Neumann_To_Dirichlet.jl")





function Data_Generate()


    N_data = 40000
    N_θ = 100
    ne=100
    seed = 123
    Random.seed!(seed)


    porder = 1
    
    θ_field = rand(Normal(0, 1.0), N_data, N_θ);
    κ_field = zeros(ne+1, ne+1, N_data)
    c_field = zeros(ne+1, ne+1, N_data)

    seq_pairs = compute_seq_pairs(N_θ)


    params = [θ_field[i, :] for i in 1:N_data]
    cmin, cmax = 100.0 , 200.0
    
    mean, std = 20.0 , 1.0
    # Define caller function
    @everywhere κ_(x::Vector{FT}) where FT<:Real = 
        Generate_sol(x, $seq_pairs, $ne, $porder, $mean, $std)

    @everywhere params = $params  

    array_of_tuples = pmap(κ_, params) # Outer dim is params iterator

    


    (κ_tuple, c_tuple) = ntuple(l->getindex.(array_of_tuples,l),2)

    

    for i in 1:N_data
        κ_field[:,:, i] = κ_tuple[i]
        c_field[:,:, i] = c_tuple[i]
    end



    npzwrite("Random_Helmholtz_high_theta_$(N_θ).npy", θ_field)
    npzwrite("Random_Helmholtz_high_K_$(N_θ).npy", κ_field)
    npzwrite("Random_Helmholtz_high_cs_$(N_θ).npy", c_field)
    
end

Data_Generate()

