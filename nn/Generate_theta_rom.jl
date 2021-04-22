using NPZ 
using LinearAlgebra
include("Box_N2D_nn_Data.jl")

function generate_random_field(θs, c_func, c_min, Δc, N_e, L=1.0)
    N_data, N_θ = size(θs)
    
    N_x = N_y = N_e + 1
    Δx = Δy = L/N_e
    xx = LinRange(0, L, N_x)

    X, Y = zeros(Float64, N_x, N_x), zeros(Float64, N_x, N_x)
    for ix = 1:N_x
        for iy = 1:N_y
            X[ix, iy] = ix*Δx
            Y[ix, iy] = iy*Δy
        end
    end
    
    seq_pairs = compute_seq_pairs(N_θ)

    cs = zeros(N_x, N_y, N_data)
    for id = 1:N_data
        for ix = 1:N_x
            for iy = 1:N_y
                x, y = X[ix, iy], Y[ix, iy]
                cs[ix, iy, id] = c_func(x, y, θs[id, :], seq_pairs, c_min, Δc)
            end
        end
    end

    return cs
end


function build_bases(cs, N_trunc=-1, acc=0.9999)

    N_x, N_y, N_data = size(cs)

    data = reshape(cs, N_x*N_y, N_data)

    # svd bases
    u, s, vh = svd(data')
    
    if N_trunc < 0
        s_sum_tot = sum(s)
        s_sum = 0.0
        for i = 1:N_data
            N_trunc = i

            s_sum += s[i]
            if s_sum > acc*s_sum_tot
                break
            end

        end
        
    end

    @info "N_trunc = ", N_trunc

    scale = mean(s[1:N_trunc])
    data_svd = u[:, 1:N_trunc] * s[1:N_trunc]/scale
    bases = vh[1:N_trunc, :]*scale

    return data_svd, bases, N_trunc

end

θs = npzread("random_direct_theta.npy")
c_func = c_func_random
c_min, Δc = 95.0, 95.0
N_e = 100
L = 1.0

cs = generate_random_field(θs, c_func, c_min, Δc, N_e, L)

data_svd, bases, N_trunc = build_bases(cs, -1, 0.99)