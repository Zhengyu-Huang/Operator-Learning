export box 

# setup the computational domain in [0,1]×[0,1]
function box(Lx::Float64, Ly::Float64, nex::Int64, ney::Int64, porder::Int64)
    
    nx, ny =  nex*porder + 1, ney*porder + 1
    nnodes, nelemss = nx*ny, nex*ney
    
    x = Array(LinRange(0.0, Lx, nx))
    y = Array(LinRange(0.0, Ly, ny))
    
    
    X, Y = MeshGrid(x, y)
    nodes = zeros(nnodes,2)
    nodes[:,1], nodes[:,2] = X'[:], Y'[:]
    
    nelems = nex*ney
    elnodes = zeros(Int64, nelems, (porder+1)^2)
    ele_id = 0
    for j = 1:ney
        for i = 1:nex 
            ele_id += 1
            n = nx*(j-1)*porder + (i-1)porder+1
            #element (i,j)
            if porder == 1
                #   4 ---- 3
                #
                #   1 ---- 2
                
                elnodes[ele_id, :] .= [n, n + 1, n + 1 + (nex + 1), n + (nex + 1)]
            elseif porder == 2
                #   4 --7-- 3
                #   8   9   6 
                #   1 --5-- 2
                elnodes[ele_id, :] .= [n, n + 2, n + 2 + 2*(2*nex+1),  n + 2*(2*nex+1), n+1, n + 2 + (2*nex+1), n + 1 + 2*(2*nex+1), n + (2*nex+1), n+1+(2*nex+1)]
            else
                error("polynomial order error, porder= ", porder)
            end
        end
    end
    
    # bottom right top left
    
    # bc_nodes is an array of nnodes × nbcs
    # 1 indicate the nnodes is in the nbcs
    nbcs = 4
    bc_nodes = zeros(Bool, nnodes, nbcs)
    bc_nodes[1:nx, 1] .= true
    bc_nodes[nx:nx:nx*ny, 2] .= true 
    bc_nodes[nx*(ny-1)+1:nx*ny, 3] .= true
    bc_nodes[1:nx:nx*(ny-1)+1, 4] .= true
    
    
    return nodes, elnodes, bc_nodes
end




