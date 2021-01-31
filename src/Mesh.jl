export box 

# setup the computational domain in [0,1]×[0,1]
function box(nex::Int64, ney::Int64, bc::Array{String, 1}, bc_func::Array{Function, 1}, 
    ω::Float64, c_func::Function, s_func::Function;
    porder::Int64 = 2, ngp::Int64 = 3, Lx::Float64=1.0, Ly::Float64=1.0)

    nx, ny =  nex*porder + 1, ney*porder + 1
    nnodes, neles = nx*ny, nex*ney
    
    x = Array(LinRange(0.0, Lx, nx))
    y = Array(LinRange(0.0, Ly, ny))
    
    
    X, Y = MeshGrid(x, y)
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X'[:], Y'[:]
    
    elements = []
    for j = 1:ney
        for i = 1:nex 
            n = nx*(j-1)*porder + (i-1)porder+1
            #element (i,j)
            if porder == 1
                #   4 ---- 3
                #
                #   1 ---- 2
                
                elnodes = [n, n + 1, n + 1 + (nex + 1), n + (nex + 1)]
            elseif porder == 2
                #   4 --7-- 3
                #   8   9   6 
                #   1 --5-- 2
                elnodes = [n, n + 2, n + 2 + 2*(2*nex+1),  n + 2*(2*nex+1), n+1, n + 2 + (2*nex+1), n + 1 + 2*(2*nex+1), n + (2*nex+1), n+1+(2*nex+1)]
            else
                error("polynomial order error, porder= ", porder)
            end
            
            coords = nodes[elnodes,:]
            push!(elements, Quad(coords, elnodes, porder, ngp, ω, c_func))
        end
    end
    
    DBC, u_g   = zeros(Int64, nnodes), zeros(Float64, nnodes)
    NBC        = zeros(Int64, nnodes)
    
    # bottom right top left
    # bottom n = 1:nx
    
    # bc_nodes is an array of nnodes × nbcs
    # 1 indicate the nnodes is in the nbcs
    nbcs = 4
    bc_nodes = zeros(Bool, nnodes, nbcs)
    bc_nodes[1:nx, 1] .= true
    bc_nodes[nx:ny:nx*ny, 2] .= true 
    bc_nodes[nx*(ny-1)+1:nx*ny, 3] .= true
    bc_nodes[1:ny:nx*(ny-1)+1, 4] .= true
    
    for bc_id = 1:4 
        for n_id = 1:nnodes
            if bc_nodes[n_id, bc_id] == true
            if bc[bc_id] == "Dirichlet"
                DBC[n_id] = -1
                u_g[n_id] = bc_func[bc_id](nodes[n_id, :]...)
            end

            if bc[bc_id] == "Neumann"
                NBC[n_id] = -1
            end
        end
        end
    end

    DBC_ele = []
    for iele = 1:neles
        ele = elements[iele]
        elnodes = ele.elnodes

        for iedge = 1:4
            if porder == 1
                loc_id = [iedge, mod1(iedge+1,4)]
            elseif porder == 2
                loc_id = [iedge, mod1(iedge+1,4), 4 + iedge]
            else
                error("porder error porder ==", porder)
            end


            if all(DBC[ele.elnodes[loc_id[:]]] .!= 0)
                push!(DBC_ele, [iele iedge])
            end
        end
    end

    DBC_ele  = vcat(DBC_ele...)



    NBC_ele = []
    for iele = 1:neles
        ele = elements[iele]
        elnodes = ele.elnodes

        for iedge = 1:4

            loc_id = getLocalEdgeNodes(ele, iedge)

            if all(NBC[ele.elnodes[loc_id[:]]] .!= 0)
                push!(NBC_ele, [iele iedge])
            end
        end
    end

    NBC_ele  = vcat(NBC_ele...)
    # @info "NBC_ele: ", NBC_ele
    ∂u∂n_ele = zeros(Float64, size(NBC_ele, 1), porder+1)

    for NBC_id = 1:size(NBC_ele, 1)
        elem_id, edge_id = NBC_ele[NBC_id, :]
        elem = elements[elem_id]
        loc_id = getLocalEdgeNodes(elem, edge_id)
        node_ids = elem.elnodes[loc_id]

        # @info "node_ids ", node_ids
        for bc_id = 1:nbcs
            # @info "bc_id = ", bc_id, bc_nodes[node_ids, bc_id]
            if all(bc_nodes[node_ids, bc_id])

                
                for k = 1:porder+1

                    # @info NBC_ele[NBC_id, :]
                    # @info bc_nodes[NBC_ele[NBC_id, :], bc_id], bc_id
                    ∂u∂n_ele[NBC_id, k] = bc_func[bc_id](nodes[node_ids[k], :]...)
                end
            end
        end
    end
    # @info "∂u∂n_ele: ", ∂u∂n_ele

    # error("stop")
    return nodes, elements, DBC, DBC_ele, u_g,  NBC, NBC_ele, ∂u∂n_ele
end