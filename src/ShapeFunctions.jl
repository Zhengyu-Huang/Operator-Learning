
using LinearAlgebra
export get2DElemShapeData, get1DElemShapeData, getShapeQuad4, getShapeQuad9
function getShapeQuad4( ξ::Array{Float64,1} )
    #   gaussian point ordering:
    #   4 ---- 3
    #
    #   1 ---- 2
    #Check the dimension of physical space
    if length(ξ) != 2
        error("2D only")
    end

    sData       = zeros(4,3)

    #Calculate shape functions
    sData[1,1] = 0.25*(1.0-ξ[1])*(1.0-ξ[2])
    sData[2,1] = 0.25*(1.0+ξ[1])*(1.0-ξ[2])
    sData[3,1] = 0.25*(1.0+ξ[1])*(1.0+ξ[2])
    sData[4,1] = 0.25*(1.0-ξ[1])*(1.0+ξ[2])
    
    #Calculate derivatives of shape functions
    sData[1,2] = -0.25*(1.0-ξ[2])
    sData[2,2] =  0.25*(1.0-ξ[2])
    sData[3,2] =  0.25*(1.0+ξ[2])
    sData[4,2] = -0.25*(1.0+ξ[2])
    
    sData[1,3] = -0.25*(1.0-ξ[1])
    sData[2,3] = -0.25*(1.0+ξ[1])
    sData[3,3] =  0.25*(1.0+ξ[1])
    sData[4,3] =  0.25*(1.0-ξ[1])
    
    return sData
end

function getShapeQuad9( ξ::Array{Float64,1} )
    #todo!! 
    #   gaussian point ordering:
    #   4 --7-- 3
    #   8   9   6 
    #   1 --5-- 2
    #Check the dimension of physical space
    if length(ξ) != 2
        error("2D only")
    end

    sData       = zeros(9,3)


    #Calculate shape functions
    sData[1,1] = ξ[1] * (ξ[1] - 1) * ξ[2] * (ξ[2] - 1) / 4.0
    sData[2,1] = ξ[1] * (ξ[1] + 1) * ξ[2] * (ξ[2] - 1) / 4.0
    sData[3,1] = ξ[1] * (ξ[1] + 1) * ξ[2] * (ξ[2] + 1) / 4.0
    sData[4,1] = ξ[1] * (ξ[1] - 1) * ξ[2] * (ξ[2] + 1) / 4.0
    sData[5,1] =  -(ξ[1] * ξ[1] - 1) * ξ[2] * (ξ[2] - 1.) / 2.0
    sData[6,1] = -ξ[1] * (ξ[1] + 1) * (ξ[2] * ξ[2] - 1.) / 2.0
    sData[7,1] = -(ξ[1] * ξ[1] - 1) * ξ[2] * (ξ[2] + 1.) / 2.0
    sData[8,1] = -ξ[1] * (ξ[1] - 1) * (ξ[2] * ξ[2] - 1.) / 2.0
    sData[9,1] = (1 - ξ[1] * ξ[1]) * (1 - ξ[2] * ξ[2])
    
    #Calculate derivative of shape functions over ξ[1] and ξ[2]

    sData[1,2] = (2 * ξ[1] - 1) * ξ[2] * (ξ[2] - 1) / 4.0
    sData[2,2] = (2 * ξ[1] + 1) * ξ[2] * (ξ[2] - 1) / 4.0
    sData[3,2] = (2 * ξ[1] + 1) * ξ[2] * (ξ[2] + 1) / 4.0
    sData[4,2] = (2 * ξ[1] - 1) * ξ[2] * (ξ[2] + 1) / 4.0
    sData[5,2] = -(2 * ξ[1]) * ξ[2] * (ξ[2] - 1.) / 2.0
    sData[6,2] = -(2 * ξ[1] + 1) * (ξ[2] * ξ[2] - 1.) / 2.0
    sData[7,2] =  -(2 * ξ[1]) * ξ[2] * (ξ[2] + 1.) / 2.0
    sData[8,2] = -(2 * ξ[1] - 1) * (ξ[2] * ξ[2] - 1.) / 2.0
    sData[9,2] = (-2 * ξ[1]) * (1 - ξ[2] * ξ[2])

                        
    
    sData[1,3] = ξ[1] * (ξ[1] - 1) * (2 * ξ[2] - 1) / 4.0
    sData[2,3] = ξ[1] * (ξ[1] + 1) * (2 * ξ[2] - 1) / 4.0
    sData[3,3] = ξ[1] * (ξ[1] + 1) * (2 * ξ[2] + 1) / 4.0
    sData[4,3] = ξ[1] * (ξ[1] - 1) * (2 * ξ[2] + 1) / 4.0
    sData[5,3] = -(ξ[1] * ξ[1] - 1) * (2 * ξ[2] - 1.) / 2.0
    sData[6,3] = -ξ[1] * (ξ[1] + 1) * (2 * ξ[2]) / 2.0
    sData[7,3] =  -(ξ[1] * ξ[1] - 1) * (2 * ξ[2] + 1.) / 2.0
    sData[8,3] =  -ξ[1] * (ξ[1] - 1) * (2 * ξ[2]) / 2.0
    sData[9,3] =   (1 - ξ[1] * ξ[1]) * (-2 * ξ[2])

    
    return sData
end

function getShapeLine2( ξ::Array{Float64,1} )
    #   gaussian point ordering:
    #   1 ---- 2
    #Check the dimension of physical space
    if length(ξ) != 1
        error("1D only")
    end

    sData       = zeros(2,2)

    #Calculate shape functions
    sData[1,1] = 0.5*(1.0-ξ[1])
    sData[2,1] = 0.5*(1.0+ξ[1])
    
    
    #Calculate derivatives of shape functions
    sData[1,2] = -0.5
    sData[2,2] =  0.5

    return sData
end

function getShapeLine3( ξ::Array{Float64,1} )
    #   gaussian point ordering:
    #   1 --3-- 2
    #Check the dimension of physical space
    if length(ξ) != 1
        error("1D only")
    end

    sData       = zeros(3,2)


    #Calculate shape functions
    sData[1,1] =  ξ[1] * (ξ[1] - 1) / 2.0
    sData[2,1] =  ξ[1] * (ξ[1] + 1) / 2.0
    sData[3,1] =-(ξ[1] * ξ[1] - 1) 
    
    
    #Calculate derivatives of shape functions
    sData[1,2] =  (2*ξ[1] - 1) / 2.0
    sData[2,2] =  (2*ξ[1] + 1) / 2.0
    sData[3,2] =  -2* ξ[1] 

    return sData
end


@doc """
    Return the Gauss quadrature points and weights in [-1,1]^n
""" -> 
function getIntegrationPoints(nPoints::Int64, ndim::Int64)
    if nPoints == 1
        q1 = [0.0]
        w1 = [2.0]
    elseif nPoints == 2
        q1 = sqrt(1. /3.)*[-1.;1.]
        w1 = [1.; 1]
    elseif nPoints == 3
        q1 = sqrt(3. /5.)*[ 0.; -1.; 1.]
        w1 = 1. /9. *[ 8.; 5.; 5.]
    elseif nPoints == 4
        q1 = [sqrt(3. / 7. - 2. /7. *sqrt(6. /5.)) ; -sqrt(3. / 7. - 2. /7. *sqrt(6. /5.)) ; sqrt(3. / 7. + 2. /7. *sqrt(6. /5.)) ; -sqrt(3. / 7. + 2. /7. *sqrt(6. /5.)) ]
        w1 = 1. / 36. * [18. + sqrt(30.); 18. + sqrt(30.); 18. - sqrt(30.); 18. - sqrt(30.)]
    end

    if ndim == 1
        return q1, w1
    end

    q2 = zeros(nPoints*nPoints, 2)
    w2 = zeros(nPoints*nPoints)
    for i = 1:nPoints
        for j = 1:nPoints
            n = (i-1)*nPoints + j
            q2[n, :] = [q1[i]; q1[j]]
            w2[n] = w1[i]*w1[j]
        end
    end
    # #@show q2
    return q2, w2
end

@doc """
    :elemCoords nnodesx2, nnodes=4 => Quad4 ; nnodes=2 => Line2
    dhdx: list of ngp shape function first order derivatives dphi/dx (nf×ndim) on the Gaussian points
    weights: list of ngp weights,  gaussian point weight and Jacobian determinant
    hs: list of ngp shape function values(nf×1) on the Gaussian points

""" ->
function get2DElemShapeData( elem_coords::Array{Float64} , npoints::Int64 = 0)

  ele_size =  size(elem_coords)
  
  #set nDim and shape function from elemType
  ndim = 2
  
  (int_coords,int_weights) = getIntegrationPoints( npoints , ndim)
#   #@show intCrds, intWghts
  dhdx = Array{Float64}[]
  weights = Float64[]
  hs = Array{Float64}[]
  for k = 1:length(int_weights)
    ξ = int_coords[k,:]
    weight = int_weights[k]
    # println(ξ)
    if ele_size[1] == 4
        sData = getShapeQuad4(ξ)
    elseif ele_size[1] == 9
        sData = getShapeQuad9(ξ) 
    else
        error("not implemented ele_size[1] = ", ele_size[1])
    end

    jac = elem_coords' * sData[:,2:end]
    
    push!(dhdx, sData[:,2:end] * inv( jac ))
    push!(weights, abs(det(jac)) * weight)
    push!(hs, sData[:,1])
  end
  

  return dhdx, weights, hs
end


@doc """
    :elemCoords nnodesx2, nnodes=3 => Line3 ; nnodes=2 => Line2
    #   1 --3-- 2 or 1 -- 2
    dhdx: list of ngp shape function first order derivatives dphi/dx (nf×ndim) on the Gaussian points
    weights: list of ngp weights,  gaussian point weight and Jacobian determinant
    hs: list of ngp shape function values(nf×1) on the Gaussian points

""" ->
function get1DElemShapeData( elem_coords::Array{Float64} , npoints::Int64 = 0)

  ele_size =  size(elem_coords)
  
  #set nDim and shape function from elemType
  ndim = 1
  
  (int_coords,int_weights) = getIntegrationPoints( npoints , ndim)

  weights = Float64[]
  hs = Array{Float64}[]
  for k = 1:length(int_weights)
    ξ = int_coords[k,:]
    weight = int_weights[k]
    
    
    if ele_size[1] == 2
        sData = getShapeLine2(ξ)
    elseif ele_size[1] == 3
        sData = getShapeLine3(ξ)
    else
        error("not implemented ele_size[1] = ", ele_size[1])
    end
    jac = elem_coords' * sData[:,2:end] #2×1
    push!(weights, sqrt(jac[1]^2 + jac[2]^2) * weight)
    push!(hs, sData[:,1])
  end
  
  
  return weights, hs
end


