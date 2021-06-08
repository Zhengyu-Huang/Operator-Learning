export MeshGrid
function MeshGrid(x::Array{Float64,1}, y::Array{Float64,1})
    nx, ny = length(x), length(y)
    X = zeros(Float64, ny, nx)
    Y = zeros(Float64, ny, nx)
    for i = 1:ny
        X[i, :] .= x
    end
    for i = 1:nx
        Y[:, i] .= y
    end

    return X, Y

end

