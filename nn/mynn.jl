# PCA Net
function FNN_Cost(ns)
    layers = length(ns) - 1
    c = 0
    for i = 1:layers
        c += 2*ns[i]*ns[i+1]
    end
    for i = 1:layers - 1
        c += ns[i+1]
    end
    return c
end
function PCA_Net_Cost(n_in, width, layers, Np)
    ns = pushfirst!([width for i = 1:layers], n_in)
    c = n_in*(2*Np-1) + (2*width - 1)*Np + FNN_Cost(ns)
    return c
end

function PARA_Net_Cost(n_in, n_y, width, layers, Np)
    ns = push!(pushfirst!([width for i = 1:layers-1], n_in+n_y), 1)
    c = n_in*(2*Np-1) + FNN_Cost(ns) * Np
    return c
end

function DeepO_Net_Cost(n_in, width, layers, Np)
    return PCA_Net_Cost(n_in, width, layers, Np)
end

function FNO_Net_Cost(df, k, layers, Np)
    #                 fourier fft/ifft        multiply    pointwise 
    d_i = d_o = 1
    return 2*Np*df*d_i + layers*(  10*df*Np*log(Np) + k*(2*df^2 - df) + df*Np + Np*(2*df^2 - df)  )  +  2*Np*d_o*df
end








function FNN_Para_Num(ns)
    layers = length(ns) - 1
    c = 0
    for i = 1:layers
        c += ns[i]*ns[i+1] + ns[i+1]
    end
    return c
end
function PCA_Para_Num(n_in, width, layers)
    ns = pushfirst!([width for i = 1:layers], n_in)
    c = FNN_Cost(ns)
    return c
end

function PARA_Para_Num(n_in, n_y, width, layers)
    ns = push!(pushfirst!([width for i = 1:layers-1], n_in+n_y), 1)
    c = FNN_Cost(ns)
    return c
end

function DeepO_Para_Num(n_in, n_y, width, layers)
    ns_branch = pushfirst!([width for i = 1:layers], n_in)
    ns_trunk = pushfirst!([width for i = 1:layers], n_y)
    c = FNN_Cost(ns_branch) + FNN_Cost(ns_trunk)
    return c
end

function FNO_Para_Num(df, k, layers)
    #                 fourier fft/ifft        multiply    pointwise 
    d_i = d_o = 1
    return df*d_i + df + df*d_o + d_o + layers*(df^2*k)
end


function compute_Para_Num(problem = "NS")
    # NS
    if problem == "NS"
        layers = 4
        n_in = 128
        n_y = 2
        k = 12^2
    # Helmholtz
    elseif problem == "Helmholtz"
        layers = 4
        n_in = 101
        n_y = 2
        k = 12^2
    elseif problem == "Solid"
        # Helmholtz
        layers = 4
        n_in = 21
        n_y = 2
        k = 12^2
    elseif problem == "Advection"
        layers = 4
        n_in = 200
        n_y = 2
        k = 12
    end


    for width in [16,64,128,256,512]
        @info "PCA-Net : ", PCA_Para_Num(n_in, width, layers)
        @info "DeepO-Net : ", DeepO_Para_Num(n_in, n_y, width, layers)
        @info "PARA-Net : ", PARA_Para_Num(n_in, n_y, width, layers)
    end


    for df in [2,4,8,16,32]
        @info "FNO : ", FNO_Para_Num(df, k, 3)
    end
end