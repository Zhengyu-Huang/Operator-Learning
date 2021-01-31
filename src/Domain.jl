import PyPlot
export Domain, visScalarField

@doc raw"""
    Domain
Date structure for the computatational domain.
- `nnodes`: Int64, number of nodes (each quadratical quad element has 9 nodes)
- `nodes`: Float64[nnodes, ndims], coordinate array of all nodes
- `neles`: number of elements 
- `elements`: a list of `neles` element arrays, each element is a struct 
- `ndims`: Int64, dimension of the problem space 
- `state`: a matrix of size `nnodes×ndims`. **Current** displacement of all nodal freedoms, `state[1:nnodes]` are for the first direction.
- `Dstate`: `nnodes×ndims`. **Previous** displacement of all nodal freedoms, `Dstate[1:nnodes]` are for the first direction.
- `LM`:  `Int64[neles][ndims]`, LM(e,d) is the global equation number (active freedom number) of element e's d th freedom, 
         
         ∘ -1 means Dirichlet
         ∘ >0 means the global equation number
- `DOF`: a matrix of size `neles×ndims`, DOF(e,d) is the global freedom (including both active and inactive DOFs) number of element e's d th freedom.
- `ID`:  a matrix of size `nnodes×ndims`. `ID(n,d)` is the equation number (active freedom number) of node n's $d$-th freedom, 
         
         ∘ -1 means fixed (time-independent) Dirichlet
         ∘ >0 means the global equation number
- `neqs`:  Int64,  number of equations, a.k.a., active freedoms
- `eq_to_dof`:  an integer vector of length `neqs`, map from to equation number (active freedom number) to the freedom number (Int64[1:nnodes] are for the first direction) 
- `dof_to_eq`:  a bolean array of size `nnodes×ndims`, map from freedom number(Int64[1:nnodes] are for the first direction) to booleans (active freedoms(equation number) are true)
- `DBC`:  Int64[nnodes, ndims], DBC[n,d] is the displacement boundary condition of node n's dth freedom,
           -1 means fixed Dirichlet boundary nodes
- `ug`:  Float64[nnodes, ndims], values for fixed(time-independent) Dirichlet boundary conditions of node n's dth freedom,
- `NBC`: Int64[nnodes, ndims], `NBC[n,d]`` is the force Neumann condition of node n's dth freedom,
           -1 means Neumann condition nodes
- `∂u∂n`:  Float64[neqs], constant (time-independent) nodal forces on these freedoms
"""

mutable struct Domain
    nnodes::Int64
    nodes::Array{Float64}
    neles::Int64
    elements::Array
    state::Array{Float64}
    LM::Array{Array{Int64}}
    DOF::Array{Array{Int64}}
    ID::Array{Int64}
    neqs::Int64
    eq_to_dof::Array{Int64}
    dof_to_eq::Array{Bool}
    DBC::Array{Int64, 1}     # Dirichlet boundary condition, -1 means the nodal value has Dirichlet condition (conrner)
    DBC_ele::Array{Int64, 2} # element face
    ug::Array{Float64, 1}    # Value for Dirichlet boundary condition
    
    NBC::Array{Int64, 1}      # Nodal boundary condition, , -1 means the nodal value has Neumann condition (conrner)
    NBC_ele::Array{Int64, 2}  # element face
    ∂u∂n_ele::Array{Float64, 2}   # Value for Neumann boundary condition element face storage

    fext::Array{Float64,1}    # external force due to ug
end



@doc raw"""
    Domain(nodes::Array{Float64}, elements::Array, ndims::Int64 = 2,
    DBC::Union{Missing, Array{Int64}} = missing, ug::Union{Missing, Array{Float64}} = missing, NBC::Union{Missing, Array{Int64}} = missing, 
    f::Union{Missing, Array{Float64}} = missing, edge_traction_data::Array{Int64,2}=zeros(Int64,0,3))
Creating a finite element domain.
- `nodes`: coordinate array of all nodes, a `nnodes × 2` matrix
- `elements`: element array. Each element is a material struct, e.g., [`PlaneStrain`](@ref). 
- `ndims`: dimension of the problem space. For 2D problems, ndims = 2. 
- `DBC`:  `nnodes × ndims` integer matrix for essential boundary conditions
    `DBC[n,d]`` is the displacement boundary condition of node `n`'s $d$-th freedom,
    
    ∘ -1: fixed (time-independent) Dirichlet boundary nodes
    ∘ -2: time-dependent Dirichlet boundary nodes
- `ug`:  `nnodes × ndims` double matrix, values for fixed (time-independent) Dirichlet boundary conditions of node `n`'s $d$-th freedom,
- `NBC`: `nnodes × ndims` integer matrix for nodal force boundary conditions.
NBC[n,d] is the force load boundary condition of node n's dth freedom,
∘ -1 means constant(time-independent) force load boundary nodes
∘ -2 means time-dependent force load boundary nodes
- `f`:  `nnodes × ndims` double matrix, values for constant (time-independent) force load boundary conditions of node n's $d$-th freedom,
- `Edge_Traction_Data`: `n × 3` integer matrix for natural boundary conditions.
`Edge_Traction_Data[i,1]` is the element id,
`Edge_Traction_Data[i,2]` is the local edge id in the element, where the force is exterted (should be on the boundary, but not required)
`Edge_Traction_Data[i,3]` is the force id, which should be consistent with the last component of the Edge_func in the Globdat
For time-dependent boundary conditions (`DBC` or `NBC` entries are -2), the corresponding `f` or `ug` entries are not used.
"""
function Domain(nodes::Array{Float64, 2}, elements::Array,
    DBC::Union{Missing, Array{Int64, 1}} = missing,   DBC_ele::Union{Missing, Array{Int64, 2}} = missing,
    ug::Union{Missing, Array{Float64, 1}} = missing, 
    NBC::Union{Missing, Array{Int64, 1}} = missing,   NBC_ele::Union{Missing, Array{Int64, 2}} = missing, 
    ∂u∂n_ele::Union{Missing, Array{Float64, 2}} = missing, s_func::Union{Missing, Function} = missing)
    
    nnodes = size(nodes,1)
    neles = size(elements,1)
    state = zeros(nnodes)
    LM = Array{Int64}[]
    DOF = Array{Int64}[]
    ID = Int64[]
    neqs = 0
    eq_to_dof = Int64[]
    dof_to_eq = zeros(Bool, nnodes)
    fext = Float64[]
    
    domain = Domain(nnodes, nodes, neles, elements, state,
    LM, DOF, ID, neqs, eq_to_dof, dof_to_eq, 
    DBC, DBC_ele, ug, NBC, NBC_ele, ∂u∂n_ele, fext)

    #set fixed(time-independent) Dirichlet boundary conditions
    setDirichletBoundary!(domain, DBC, ug)
    #set constant(time-independent) force load boundary conditions
    setNeumannBoundary!(domain, NBC, NBC_ele, ∂u∂n_ele)

    setBodyForce!(domain, s_func)

    domain
end




@doc """
    setConstantDirichletBoundary!(self::Domain, DBC::Array{Int64}, ug::Array{Float64})
Bookkeepings for time-independent Dirichlet boundary conditions. Only called once in the constructor of `domain`. 
It updates the fixed (time-independent Dirichlet boundary) state entries and builds both LM and DOF arrays.
- `self`: Domain
- `DBC`:  Int64[nnodes, ndims], DBC[n,d] is the displacement boundary condition of node n's dth freedom,
        
  ∘ -1 means fixed(time-independent) Dirichlet boundary nodes
  ∘ -2 means time-dependent Dirichlet boundary nodes
- `ug`:  Float64[nnodes, ndims], values for fixed (time-independent) Dirichlet boundary conditions of node n's dth freedom,
""" -> 
function setDirichletBoundary!(domain::Domain, DBC::Array{Int64}, ug::Array{Float64})

    # ID(n,d) is the global equation number of node n's dth freedom, 
    # -1 means fixed (time-independent) Dirichlet
    # -2 means time-dependent Dirichlet

    nnodes = domain.nnodes
    neles, elements = domain.neles, domain.elements
    #ID = zeros(Int64, nnodes, ndims) .- 1

    ID = copy(DBC)

    eq_to_dof, dof_to_eq = Int64[], zeros(Bool, nnodes)
    neqs = 0
      for inode = 1:nnodes
          if (DBC[inode] == 0)
              neqs += 1
              ID[inode] = neqs
              push!(eq_to_dof,inode)
              dof_to_eq[inode] = true
          elseif (DBC[inode] == -1)
              #update state fixed (time-independent) Dirichlet boundary conditions
              domain.state[inode] = ug[inode]
          end
      end


    domain.ID, domain.neqs, domain.eq_to_dof, domain.dof_to_eq = ID, neqs, eq_to_dof, dof_to_eq


    # LM(e,d) is the global equation number of element e's d th freedom
    LM = Array{Array{Int64}}(undef, neles)
    for iele = 1:neles
      el_nodes = getNodes(elements[iele])
      ieqns = ID[el_nodes, :][:]
      LM[iele] = ieqns
    end
    domain.LM = LM

    # DOF(e,d) is the global dof number of element e's d th freedom

    DOF = Array{Array{Int64}}(undef, neles)
    for iele = 1:neles
      el_nodes = getNodes(elements[iele])
        DOF[iele] = el_nodes
    end
    domain.DOF = DOF
    
end


@doc """
Bookkeepings for time-independent Nodal force boundary conditions. Only called once in the constructor of `domain`. 
It updates the fixed (time-independent Nodal forces) state entries and builds both LM and DOF arrays.
- `self`: Domain
- `NBC`:  Int64[nnodes, ndims], NBC[n,d] is the displacement boundary condition of node n's dth freedom,
        
    ∘ -1 means fixed (time-independent) Nodal force freedoms
    ∘ -2 means time-dependent Nodal force freedoms
- `∂u∂n`:  Float64[nnodes, ndims], values for fixed (time-independent) Neumann boundary conditions of node n's dth freedom,
#The name is misleading
""" 

function setNeumannBoundary!(self::Domain, NBC::Array{Int64}, NBC_ele::Array{Int64}, ∂u∂n_ele::Array{Float64})
    ID = self.ID
    fext = zeros(Float64, self.neqs)
    # ID(n,d) is the global equation number of node n's dth freedom, -1 means no freedom
    
    # loop all edges
    nNBC = size(NBC_ele, 1)
    for nNBC_id = 1:nNBC
        eleid, eid  = NBC_ele[nNBC_id, :]
        elem = self.elements[eleid]
        edge_local_nodes = getLocalEdgeNodes(elem, eid)
        edge_nodes = elem.elnodes[edge_local_nodes]
        fext_local = computeLoad(elem, eid, ∂u∂n_ele[nNBC_id, :])

        # @show eleid, eid, fext_local

        for i = 1:length(fext_local)
            inode = edge_nodes[i]
            # @info "inode, ID[inode] :", inode, ID[inode]
            if  ID[inode] > 0
                fext[ID[inode]] += fext_local[i]
            end
        end
    end
    
    self.fext = fext
end







@doc raw"""
    getBodyForce(domain::Domain, globdat::GlobalData, time::Float64)
Computes the body force vector $F_\mathrm{body}$ of length `neqs`
- `globdat`: GlobalData
- `domain`: Domain, finite element domain, for data structure
- `Δt`:  Float64, current time step size
"""
function setBodyForce!(domain::Domain, s_func::Function)
    
    fbody = zeros(Float64, domain.neqs)
    neles = domain.neles

    # Loop over the elements in the elementGroup
    fe = zeros(Float64, domain.elements[1].ngp^2)
    for iele  = 1:neles
        element = domain.elements[iele]
  
        gauss_pts = getGaussPoints(element.coords, element.h, element.ngp^2)

        for i = 1:length(fe)
            fe[i] = s_func(gauss_pts[i,1], gauss_pts[i,2])
        end
        

        f = getBodyForce(element, fe)

        # Assemble in the global array
        el_eqns = getEqns(domain, iele)
        el_eqns_active = (el_eqns .>= 1)
        fbody[el_eqns[el_eqns_active]] += f[el_eqns_active]
    end
  
    domain.fext .+= fbody
end



@doc """
    getCoords(domain::Domain, el_nodes::Array{Int64})
Get the coordinates of several nodes (possibly in one element)
- `domain`: Domain
- `el_nodes`: Int64[n], node array
Return: Float64[n, ndims], the coordinates of these nodes
"""
function getCoords(domain::Domain, el_nodes::Array{Int64})
    return domain.nodes[el_nodes, :]
end

@doc """
    getDofs(domain::Domain, iele::Int64)   
Get the global freedom numbers of the element
- `domain`: Domain
- `iele`: Int64, element number
Return: Int64[], the global freedom numbers of the element (ordering in local element ordering)
""" 
function getDofs(domain::Domain, iele::Int64)    
    return domain.DOF[iele]
end

@doc """
    getNGauss(domain::Domain)
Gets the total number of Gauss quadrature points. 
"""
function getNGauss(domain::Domain)
    ng = 0
    for e in domain.elements
        ng += length(e.weights)
    end
    ng
end

@doc """
    getEqns(domain::Domain, iele::Int64)
Gets the equation numbers (active freedom numbers) of the element. 
This excludes both the time-dependent and time-independent Dirichlet boundary conditions. 
""" 
function getEqns(domain::Domain, iele::Int64)
    return domain.LM[iele]
end


@doc """
    getState(domain::Domain, el_dofs::Array{Int64})
Get the displacements of several nodes (possibly in one element)
- `domain`: Domain
- `el_nodes`: Int64[n], node array
Return: Float64[n, ndims], the displacements of these nodes
""" 
function getState(domain::Domain, el_dofs::Array{Int64})
    return domain.state[el_dofs]
end



@doc raw"""
    getGaussPoints(domain::Domain)
Returns all Gauss points as a $n_g\times 2$ matrix, where $n_g$ is the total number of Gauss points.
"""
function getGaussPoints(domain::Domain)
    v = []
    for e in domain.elements
        vg = getGaussPoints(e) 
        push!(v, vg)
    end 
    vcat(v...)
end



@doc raw"""
    getElems(domain::Domain)
Returns the element connectivity matrix $n_e \times 4$. This function implicitly assumes that all elements are quadrilateral.
"""
function getElems(domain::Domain)
    elem = zeros(Int64, domain.neles, 4)
    for (k,e) in enumerate(domain.elements)
        elem[k,:] = e.elnodes
    end
    return elem
end


function visScalarField(domain::Domain, state::Array{Float64, 1}; shading= "gouraud", cmap="viridis")
    vmin, vmax = minimum(state), maximum(state)
    for e in domain.elements

        eXY = e.coords
        eC = state[e.elnodes]
        porder = e.porder

        if porder == 1
            eX = [eXY[1, 1] eXY[2, 1]; eXY[4, 1] eXY[3, 1]]
            eY = [eXY[1, 2] eXY[2, 2]; eXY[4, 2] eXY[3, 2]]
            eC = [eC[1] eC[2]; eC[4] eC[3]]
        elseif porder == 2
            eX = [eXY[1, 1] eXY[5, 1] eXY[2, 1]; eXY[8, 1] eXY[9, 1] eXY[6, 1];  eXY[4, 1] eXY[7, 1] eXY[3, 1]]
            eY = [eXY[1, 2] eXY[5, 2] eXY[2, 2]; eXY[8, 2] eXY[9, 2] eXY[6, 2];  eXY[4, 2] eXY[7, 2] eXY[3, 2]]
            eC = [eC[1] eC[5] eC[2]; eC[8] eC[9] eC[6];  eC[4] eC[7] eC[3]]
        end

        # @info "eX, eY, eC: ", eX, eY, eC

        PyPlot.pcolormesh(eX, eY, eC, shading = shading, cmap = cmap, vmin = vmin, vmax = vmax)
        
    end
    PyPlot.colorbar()

end
