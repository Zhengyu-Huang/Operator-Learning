export Quad


"""
Quad
Constructs a finite strain element. 
- `eledim`: spatial dimension of the element (default = 2).
- `mat`: constitutive law, a length `#elem` vector of materials such as [`PlaneStress`](@ref)
- `elnodes`: the node indices in this finite element, an integer array 
- `props`: property dictionary 
- `coords`: coordinates of the vertices of the element
- `dhdx`, `weights`, `hs`: data for integral 
- `stress`: stress at each quadrature points
# Example
```julia
#   Local degrees of freedom 
#   4 ---- 3
#
#   1 ---- 2
or
#   4 --7-- 3
#   8   9   6 
#   1 --5-- 2
```
"""
mutable struct Quad
    porder::Int64 
    ngp::Int64
    elnodes::Array{Int64}   # the node indices in this finite element
    coords::Array{Float64}
    h::Array{Array{Float64}}
    dhdx::Array{Array{Float64}}  # 4nPointsx2 matrix
    weights::Array{Float64} 
    hs::Array{Array{Float64}}

    ω_div_c_sq::Array{Float64}
end

Base.show(io::IO, z::Quad) = print(io, "Quad with $(z.ngp) Gauss quadrature points.")


"""
Quad(coords::Array{Float64}, elnodes::Array{Int64}, props::Dict{String, Any}, ngp::Int64=2)
"""
function Quad(coords::Array{Float64}, elnodes::Array{Int64}, porder::Int64, ngp::Int64, 
              ω::Float64, c_func::Function, )
    dhdx, weights, hs = get2DElemShapeData( coords, ngp )
    
    gps = getGaussPoints(coords, hs, ngp^2)

    ω_div_c_sq = fill!(zeros(Float64, ngp^2), ω^2)
    for i = 1:ngp^2
        ω_div_c_sq[i] /= c_func(gps[i,1], gps[i,2])^2
    end

    Quad(porder, ngp, elnodes, coords, hs, dhdx, weights, hs, ω_div_c_sq)
end
"""
∫ ∇ u ∇ϕ - ω^2/c^2 u
"""
function getStiffAndForce(self::Quad, us::Array{Float64})
    
    ndofs = dofCount(self); 
    nnodes = length(self.elnodes)
    fint = zeros(Float64, ndofs)
    stiff = zeros(Float64, ndofs,ndofs)
    ω_div_c_sq = self.ω_div_c_sq
    for k = 1:length(self.weights)
        dh = self.dhdx[k]

        h = self.h[k]
        
        u = us' * h
        
        dus = us' * dh 

        fint += (dh * dus' -  ω_div_c_sq[k] * u * h) * self.weights[k] # 1x8
    
        stiff += (dh * dh' -  ω_div_c_sq[k] * h * h') * self.weights[k] # 8x8
    end
    
    return fint, stiff
end

function getLocalEdgeNodes(elem::Quad, edge_id::Int64)
    porder = elem.porder

    if porder == 1
        loc_node_ids = [edge_id, mod1(edge_id+1,4)]
    elseif porder == 2
        loc_node_ids = [edge_id, mod1(edge_id+1,4), 4 + edge_id]
    else
        error("porder error porder ==", porder)
    end
end

"""
Force load on one edge of the plate, 
The edge is [0,L], which is discretized to ne elements, with porder-th order polynomials
type: a string of load type, which can be "constant", or "Gaussian"
args: an array, 
for Constant load, it has p1 and p2, in tangential and normal direction
for "Gaussian" load, it has p, x0, and σ, the force is in the normal direction
"""
function computeLoad(elem::Quad, e::Int64, ∂u∂n_ele::Array{Float64,1})
    
    porder = elem.porder
    
    elem_coords = zeros(Float64, porder + 1, 2)
    
    
    loc_node_ids = getLocalEdgeNodes(elem, e)
    
    
    edge_coords = elem.coords[loc_node_ids, :]
    
    f = zeros(Float64, porder+1)
    
    ngp = elem.ngp
    
    #return list 
    weights, hs = get1DElemShapeData( edge_coords, ngp)  
    # @info "computeLoad: ", ∂u∂n_ele' , hs[1]
    for igp = 1:ngp
        ∂u∂n_g = ∂u∂n_ele' * hs[igp]   #x-coordinates of element Gaussian points
        f += ∂u∂n_g * hs[igp] * weights[igp]
    end
    
    # @info "f is ", f
    return f
end

"""
Force load on one edge of the plate, 
The edge is [0,L], which is discretized to ne elements, with porder-th order polynomials
type: a string of load type, which can be "constant", or "Gaussian"
args: an array, 
for Constant load, it has p1 and p2, in tangential and normal direction
for "Gaussian" load, it has p, x0, and σ, the force is in the normal direction
"""
function compute∂u∂n(elem::Quad, edge_id::Int64, u::Array{Float64, 1})
    loc_node_ids = getLocalEdgeNodes(elem, edge_id)
    coords = elem.coords

    porder = elem.porder
    ∂u∂n_ele = zeros(Float64, porder+1)
    # ξ coordinates of each nodal point
    
    ξ_coords = [-1.0 -1.0;
    1.0  -1.0;
    1.0   1.0;
    -1.0  1.0;
    0.0  -1.0;
    1.0   0.0;
    0.0   1.0;
    -1.0  0.0;
    0.0   0.0]
    
    ∂u∂n = zeros(Float64, porder+1)
    for k = 1:porder+1
        
        ξ = ξ_coords[loc_node_ids[k], :]
        # println(ξ)
        if porder == 1
            sData = getShapeQuad4(ξ)
        elseif porder == 2
            sData = getShapeQuad9(ξ) 
        else
            error("not implemented porder = ", porder)
        end
        #(∂u/∂x, ∂u/∂y) = (∂u/∂ξ1, ∂u/∂ξ2) [∂(x,y)/∂(ξ1,ξ2)]⁻¹
        dhdx = sData[:,2:end]
        jac = coords' * dhdx

        # tagent = (edge_id == 1 || edge_id == 3 ? jac[1, :] : jac[2, :])
        if edge_id == 1
            tangent = jac[1, :]
        elseif edge_id == 2
            tangent = jac[2, :]
        elseif edge_id == 3
            tangent = -jac[1, :]
        elseif edge_id == 4
            tangent = -jac[2, :]
        else
            error("local edge id is ", edge_id)
        end

        unit_normal = [tangent[2] ; -tangent[1]]/sqrt(tangent[1]^2 + tangent[2]^2)

        ∂u∂n_ele[k] = u' * dhdx / jac * unit_normal
    end

    return ∂u∂n_ele
end

""" 
    getNodes(elem::Quad)
Alias for `elem.elnodes`
"""
function getNodes(elem::Quad)
    return elem.elnodes
end

"""
    getGaussPoints(elem::Quad)
Returns the Gauss quadrature nodes of the element in the undeformed domain
"""
function getGaussPoints(coords, hs, ngp_2d)
    gnodes = zeros(ngp_2d,2)
    for k = 1:ngp_2d
        gnodes[k,:] = coords' * hs[k] 
    end
    return gnodes
end

"""
    getEdgeGaussPoints(elem::Quad, edge_id::Int64)
```
    The element nodes are ordered as 
    #   4 ---- 3             #   4 --7-- 3
    #                        #   8   9   6 
    #   1 ---- 2             #   1 --5-- 2
    for porder=1     or          porder=2
    edge 1, 2, 3, 4 are (1,2), (2,3), (3,4), (4,1)
                    are (1,2,5), (2,3,6), (3,4,7), (4,1,8)
```
Returns the Gauss quadrature nodes of the element on its edge_id-th edge in the undeformed domain
"""
function getEdgeGaussPoints(elem::Quad, edge_id::Int64)
    n = length(elem.elnodes)
    ngp = Int64(sqrt(length(elem.weights)))

    @assert(n == 4 || n == 9)

    n1, n2 = edge_id, ((edge_id+1)==5 ? 1 : edge_id+1)
    loc_node_ids = (n == 4 ? [n1, n2] : [n1, n2, edge_id+4])

    x = elem.coords[loc_node_ids, :]

    gnodes = zeros(ngp,2)

    _, hs = get1DElemShapeData(x, ngp)  

    gnodes = zeros(ngp,2)   
    for igp = 1:ngp
        gnodes[igp,:] = x' * hs[igp] 
    end

    return gnodes
end


function dofCount(elem::Quad)
    return length(elem.elnodes)
end

function getBodyForce(elem::Quad, fvalue::Array{Float64,1})
    n = dofCount(elem)
    fbody = zeros(Float64,n)

    nnodes = length(elem.elnodes)
    for k = 1:length(elem.weights)
        fbody[1:nnodes] += elem.hs[k] * fvalue[k,1] * elem.weights[k]
    end
    return fbody
end