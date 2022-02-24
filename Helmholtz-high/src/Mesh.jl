export box, ring

# setup the computational domain in [0,Lx]×[0,Ly]
# the node and element order is from the bottom left to the top right
#  ...
# nx+1 nx+2    ... 2nx
# 1    2       ... nx
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




# setup the computational domain in [0,Lx]×[0,Ly]
function ring(r::Float64, R::Float64, neθ::Int64, ner::Int64, porder::Int64)
    
    nθ, nr =  neθ*porder, ner*porder + 1
    nnodes, nelemss = nθ*nr, neθ*ner
    
    θ_1d = Array(LinRange(0.0, -2π, nθ+1))[1:nθ]
    # element in counterclock order
    r_1d = Array(LinRange(r, R, nr))
    
    nodes = zeros(nnodes,2)
    n_id = 0
    for j = 1:nr
        for i = 1:nθ 
            n_id += 1
            nodes[n_id,1], nodes[n_id,2] = r_1d[j]*cos(θ_1d[i]), r_1d[j]*sin(θ_1d[i])
        end
    end

    nelems = neθ*ner
    elnodes = zeros(Int64, nelems, (porder+1)^2)
    ele_id = 0
    for j = 1:ner
        for i = 1:neθ 
            ele_id += 1
            n = nθ*(j-1)*porder + (i-1)porder+1
            #element (i,j)
            if porder == 1
                #   4 ---- 3
                #
                #   1 ---- 2
                np1 = (i < neθ ? n + 1 : nθ*(j-1)*porder + 1)
                elnodes[ele_id, :] .= [n, np1, np1 + neθ, n + neθ]
            elseif porder == 2
                #   4 --7-- 3
                #   8   9   6 
                #   1 --5-- 2
                np2 = (i < neθ ? n + 2 : nθ*(j-1)*porder + 1)
                elnodes[ele_id, :] .= [n, np2, np2+2*(2*neθ),  n + 2*(2*neθ), n+1, np2 + (2*neθ), n+1+2*(2*neθ), n+(2*neθ), n+1+(2*neθ)]
            else
                error("polynomial order error, porder= ", porder)
            end
        end
    end
    # inner outer bc
    
    # bc_nodes is an array of nnodes × nbcs
    # 1 indicate the nnodes is in the nbcs
    nbcs = 2
    bc_nodes = zeros(Bool, nnodes, nbcs)
    bc_nodes[1:nθ, 1] .= true
    bc_nodes[nθ*(nr-1)+1:nθ*nr, 2] .= true
    
    return nodes, elnodes, bc_nodes
end

# setup the computational domain in [0,Lx]×[0,Ly]
function lshape(Lx::Float64, Ly::Float64, nex::Int64, ney::Int64, porder::Int64)
    
    @assert(nex % 2 == 0 && ney % 2 == 0)
    nex2, ney2 = div(nex,2), div(ney, 2)
    nx, ny = nex*porder + 1, ney*porder + 1
    nx2, ny2 = nex2*porder + 1, ney2*porder + 1
    
    nnodes, nelemss = nx*ny2 + nx2*ny2 - nx2, 3*nex2*ney2
    
    x = Array(LinRange(0.0, Lx, nx))
    y = Array(LinRange(0.0, Ly, ny))
    # bottom block
    X, Y = MeshGrid(x, y[1:ny2])
    nodes = zeros(nnodes,2)
    nodes[1:nx*ny2,1], nodes[1:nx*ny2,2] = X'[:], Y'[:]
    # top block
    X, Y = MeshGrid(x[1:nx2], y[ny2+1:end])
    nodes[nx*ny2+1:end,1], nodes[nx*ny2+1:end,2] = X'[:], Y'[:]
    
    nelems = nex*ney - nex2*ney2
    elnodes = zeros(Int64, nelems, (porder+1)^2)
    ele_id = 0
    for j = 1:ney2
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
    
    j = ney2+1
    for i = 1:nex2
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
            elnodes[ele_id, :] .= [n, n + 2, n + 2 + (3*nex+2),  n + (3*nex+2), n+1, n + 2 + (2*nex+1), n + 1 + (3*nex+2), n + (2*nex+1), n+1+(2*nex+1)]
        else
            error("polynomial order error, porder= ", porder)
        end
    end
    
    
    for j = ney2+2:ney
        for i = 1:nex2 
            ele_id += 1
            n = nx*ny2 + (porder-1)*nx2 + 1 + nx2*(j-ney2-2)*porder + (i-1)porder
            #element (i,j)
            if porder == 1
                #   4 ---- 3
                #
                #   1 ---- 2
                
                elnodes[ele_id, :] .= [n, n + 1, n + 1 + (nex2 + 1), n + (nex2 + 1)]
            elseif porder == 2
                #   4 --7-- 3
                #   8   9   6 
                #   1 --5-- 2
                elnodes[ele_id, :] .= [n, n + 2, n + 2 + 2*(2*nex2+1),  n + 2*(2*nex2+1), n+1, n + 2 + (2*nex2+1), n + 1 + 2*(2*nex2+1), n + (2*nex2+1), n+1+(2*nex2+1)]
            else
                error("polynomial order error, porder= ", porder)
            end
        end
    end
    
    # bottom right1 top1 right2 top2 left
    
    # bc_nodes is an array of nnodes × nbcs
    # 1 indicate the nnodes is in the nbcs
    nbcs = 6
    bc_nodes = zeros(Bool, nnodes, nbcs)
    bc_nodes[1:nx, 1] .= true
    bc_nodes[nx:nx:nx*ny2, 2] .= true 
    bc_nodes[nx*ny2-nx2+1:nx*ny2, 3] .= true
    bc_nodes[[nx*ny2-nx2+1; nx*ny2-nx2+nx+1:nx2:nnodes], 4] .= true
    bc_nodes[nnodes-nx2+1: nnodes, 5] .= true
    bc_nodes[[1:nx:nx*(ny2-1)+1; nx*ny2+1:nx2:nnodes], 6] .= true
    
    return nodes, elnodes, bc_nodes
end