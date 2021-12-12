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

function PARA_Net_Cost(n_in, width, layers, Np)
    ns = push!(pushfirst!([width for i = 1:layers-1], n_in+2), 1)
    c = n_in*(2*Np-1) + FNN_Cost(ns) * Np
    return c
end

function DeepO_Net_Cost(n_in, width, layers, Np)
    return PCA_Net_Cost(n_in, width, layers, Np)
end

function FNO_Net_Cost(df, k, layers, Np)
    #                 fourier fft/ifft        multiply    pointwise 
    return Np*df + (2df * 5 * Np * log2(Np) + 2*df*k*k  + Np + Np + Np)*layers +  Np*(2df-1)
end