@everywhere using Random, Distributions, NPZ
@everywhere include("PlatePull.jl")





function Data_Generate()

    seed = 123
    Random.seed!(seed)

    N_data = 40000
    
    
    porder = 2
    N_θ = 100



    θ_field = rand(Normal(0, 1.0), N_data, N_θ);
    XY, σ, Fn = GenerateData(θ_field[1, :], 2; plot=true)

    σ_field = zeros(length(σ), N_data)
    Fn_field = zeros(length(Fn), N_data)

    params = [θ_field[i, :] for i in 1:N_data]

    # Define caller function
    @everywhere f_sigma(x::Vector{FT}) where FT<:Real = 
        GenerateData(x, $porder)

    @everywhere params = $params  

    array_of_tuples = pmap(f_sigma, params) # Outer dim is params iterator

    (XY_tuple, σ_tuple, Fn_tuple) = ntuple(l->getindex.(array_of_tuples,l),3)

    

    for i in 1:N_data
        σ_field[:, i] = σ_tuple[i]
        Fn_field[:, i] = Fn_tuple[i]
    end



    npzwrite("Random_UnitCell_theta_$(N_θ).npy", θ_field)
    npzwrite("Random_UnitCell_sigma_$(N_θ).npy", σ_field)
    npzwrite("Random_UnitCell_Fn_$(N_θ).npy", Fn_field)
    npzwrite("Random_UnitCell_XY_$(N_θ).npy", XY)
    
end

Data_Generate()

