import PyPlot
export Domain, visScalarField

@doc raw"""
Domain
Date structure for the computatational domain.
    - `nnodes`: Int64, number of nodes (each quadratical quad element has 9 nodes)
    - `nodes`: Float64[nnodes, ndims], coordinate array of all nodes
    - `nelems`: number of elements 
    - `elements`: a list of `nelems` element arrays, each element is a struct 
    - `ndims`: Int64, dimension of the problem space 
    - `state`: a matrix of size `nnodes×ndims`. **Current** displacement of all nodal freedoms, `state[1:nnodes]` are for the first direction.
    - `Dstate`: `nnodes×ndims`. **Previous** displacement of all nodal freedoms, `Dstate[1:nnodes]` are for the first direction.
    - `LM`:  `Int64[nelems][ndims]`, LM(e,d) is the global equation number (active freedom number) of element e's d th freedom, 
    
    ∘ -1 means Dirichlet
    ∘ >0 means the global equation number
    - `DOF`: a matrix of size `nelems×ndims`, DOF(e,d) is the global freedom (including both active and inactive DOFs) number of element e's d th freedom.
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
    nelems::Int64
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
    DBC::Array{Int64, 1},   DBC_ele::Array{Int64, 2},
    ug::Array{Float64, 1}, 
    NBC::Array{Int64, 1},   NBC_ele::Array{Int64, 2}, 
    ∂u∂n_ele::Array{Float64, 2}, s_func::Function)
    
    nnodes = size(nodes,1)
    nelems = size(elements,1)
    state = zeros(nnodes)
    LM = Array{Int64}[]
    DOF = Array{Int64}[]
    ID = Int64[]
    neqs = 0
    eq_to_dof = Int64[]
    dof_to_eq = zeros(Bool, nnodes)
    fext = Float64[]
    
    domain = Domain(nnodes, nodes, nelems, elements, state,
    LM, DOF, ID, neqs, eq_to_dof, dof_to_eq, 
    DBC, DBC_ele, ug, NBC, NBC_ele, ∂u∂n_ele, fext)

    #set fixed(time-independent) Dirichlet boundary conditions
    setDirichletBoundary!(domain, DBC, ug)
    #set constant(time-independent) force load boundary conditions
    setNeumannBoundary!(domain, NBC, NBC_ele, ∂u∂n_ele)

    setBodyForce!(domain, s_func)

    domain
end


function Domain(nodes::Array{Float64, 2}, elnodes::Array{Int64, 2},
        bc_nodes::Array{Bool, 2}, bc_types::Array{String, 1}, bc_funcs::Array{Function, 1},
        porder::Int64 = 2, ngp::Int64 = 3; 
        ω::Float64 = 0.0, 
        c_func::Function = (x,y)->1.0, 
        s_func::Function = (x,y)->0.0)
    
    nnodes = size(nodes,1)
    nelems = size(elnodes,1)
    # create elements
    elements = []
    for elem_id = 1:nelems
        coords = nodes[elnodes[elem_id,:],:]
        push!(elements, Quad(coords, elnodes[elem_id,:], porder, ngp, ω, c_func))
    end
    
    DBC, DBC_ele, ug, NBC, NBC_ele, ∂u∂n_ele = PreProcessBC(nodes, elements, bc_nodes, bc_types, bc_funcs)
    
    
    state = zeros(nnodes)
    LM = Array{Int64}[]
    DOF = Array{Int64}[]
    ID = Int64[]
    neqs = 0
    eq_to_dof = Int64[]
    dof_to_eq = zeros(Bool, nnodes)
    fext = Float64[]
    
    domain = Domain(nnodes, nodes, nelems, elements, state,
    LM, DOF, ID, neqs, eq_to_dof, dof_to_eq, 
    DBC, DBC_ele, ug, NBC, NBC_ele, ∂u∂n_ele, fext)
    
    #set fixed(time-independent) Dirichlet boundary conditions
    setDirichletBoundary!(domain, DBC, ug)
    #set constant(time-independent) force load boundary conditions
    setNeumannBoundary!(domain, NBC, NBC_ele, ∂u∂n_ele)
    
    setBodyForce!(domain, s_func)
    
    domain
end

function PreProcessBC(nodes::Array{Float64, 2}, elements::Array, 
    bc_nodes::Array{Bool, 2}, bc_types::Array{String, 1}, bc_funcs::Array{Function, 1},)
    
    nnodes, nbcs = size(bc_nodes)
    nelems = length(elements)
    porder = elements[1].porder
    # preprocess boundary conditions
    DBC, ug   = zeros(Int64, nnodes), zeros(Float64, nnodes)
    NBC        = zeros(Int64, nnodes)
    
    for bc_id = 1:nbcs
        for n_id = 1:nnodes
            if bc_nodes[n_id, bc_id] == true
                if bc_types[bc_id] == "Dirichlet"
                    DBC[n_id] = -1
                    ug[n_id] = bc_funcs[bc_id](nodes[n_id, :]...)
                end
                
                if bc_types[bc_id] == "Neumann"
                    NBC[n_id] = -1
                end
            end
        end
    end
    
    DBC_ele = []
    for elem_id = 1:nelems
        elem = elements[elem_id]
        elnodes = elem.elnodes
        
        for edge_id = 1:4
            loc_node_ids = getLocalEdgeNodes(elem, edge_id)
            node_ids =  elem.elnodes[loc_node_ids]
            
            for bc_id = 1:nbcs
                if all(bc_nodes[node_ids, bc_id])
                    push!(DBC_ele, [elem_id edge_id bc_id])
                    break
                end
            end
        end
    end
    
    DBC_ele  = vcat(DBC_ele...)
    
    NBC_ele = []
    for elem_id = 1:nelems
        elem = elements[elem_id]
        elnodes = elem.elnodes
        
        for edge_id = 1:4
            
            loc_node_ids = getLocalEdgeNodes(elem, edge_id)
            node_ids =  elem.elnodes[loc_node_ids]
            
            for bc_id = 1:nbcs
                if all(bc_nodes[node_ids, bc_id])
                    push!(NBC_ele, [elem_id edge_id bc_id])
                    break
                end
            end
        end
    end
    
    NBC_ele  = vcat(NBC_ele...)
    ∂u∂n_ele = zeros(Float64, size(NBC_ele, 1), porder+1)
    
    for NBC_id = 1:size(NBC_ele, 1)
        elem_id, edge_id, bc_id = NBC_ele[NBC_id, :]
        elem = elements[elem_id]
        loc_node_ids = getLocalEdgeNodes(elem, edge_id)
        node_ids = elem.elnodes[loc_node_ids]
        
        for k = 1:porder+1
            ∂u∂n_ele[NBC_id, k] = bc_funcs[bc_id](nodes[node_ids[k], :]...)
        end
    end
    
    return DBC, DBC_ele, ug, NBC, NBC_ele, ∂u∂n_ele
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
    nelems, elements = domain.nelems, domain.elements
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
    LM = Array{Array{Int64}}(undef, nelems)
    for elem_id = 1:nelems
        el_nodes = getNodes(elements[elem_id])
        ieqns = ID[el_nodes, :][:]
        LM[elem_id] = ieqns
    end
    domain.LM = LM
    
    # DOF(e,d) is the global dof number of element e's d th freedom
    
    DOF = Array{Array{Int64}}(undef, nelems)
    for elem_id = 1:nelems
        el_nodes = getNodes(elements[elem_id])
        DOF[elem_id] = el_nodes
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
    nelems = domain.nelems

    # Loop over the elements in the elementGroup
    fe = zeros(Float64, domain.elements[1].ngp^2)
    for elem_id  = 1:nelems
        element = domain.elements[elem_id]
  
        gauss_pts = getGaussPoints(element.coords, element.h, element.ngp^2)

        for i = 1:length(fe)
            fe[i] = s_func(gauss_pts[i,1], gauss_pts[i,2])
        end
        

        f = getBodyForce(element, fe)

        # Assemble in the global array
        el_eqns = getEqns(domain, elem_id)
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
    getDofs(domain::Domain, elem_id::Int64)   
Get the global freedom numbers of the element
- `domain`: Domain
- `elem_id`: Int64, element number
Return: Int64[], the global freedom numbers of the element (ordering in local element ordering)

""" 
function getDofs(domain::Domain, elem_id::Int64)    
    return domain.DOF[elem_id]
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
    getEqns(domain::Domain, elem_id::Int64)
Gets the equation numbers (active freedom numbers) of the element. 
This excludes both the time-dependent and time-independent Dirichlet boundary conditions. 

""" 
function getEqns(domain::Domain, elem_id::Int64)
    return domain.LM[elem_id]
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
    elem = zeros(Int64, domain.nelems, 4)
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
