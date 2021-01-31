using SparseArrays
export assembleStiffAndForce, solve!

function assembleStiffAndForce(domain::Domain)
    Fint = zeros(Float64, domain.neqs)
  # K = zeros(Float64, domain.neqs, domain.neqs)
    ii = Int64[]; jj = Int64[]; vv = Float64[]
    neles = domain.neles

  # Loop over the elements in the elementGroup
    for iele  = 1:neles
        element = domain.elements[iele]

    # Get the element nodes
        el_nodes = getNodes(element)

    # Get the element nodes equation numbers
        el_eqns = getEqns(domain, iele)
    
    # Get the element nodes dof numbers
        el_dofs = getDofs(domain, iele)

        el_state  = getState(domain, el_dofs)


    # Get the element contribution by calling the specified action
        fint, stiff  = getStiffAndForce(element, el_state)

    # Assemble in the global array
        el_eqns_active = el_eqns .>= 1
    # K[el_eqns[el_eqns_active], el_eqns[el_eqns_active]] += stiff[el_eqns_active,el_eqns_active]
        Slocal = stiff[el_eqns_active,el_eqns_active]
        Idx = el_eqns[el_eqns_active]
        for i = 1:length(Idx)
            for j = 1:length(Idx)
                push!(ii, Idx[i])
                push!(jj, Idx[j])
                push!(vv, Slocal[i,j])
            end
        end
        Fint[el_eqns[el_eqns_active]] += fint[el_eqns_active]
    end
    Ksparse = sparse(ii, jj, vv, domain.neqs, domain.neqs)

    return Fint, Ksparse
end

# Fint = Fext
function solve!(domain::Domain)
    # @info "state: ", domain.state
    Fint, Ksparse = assembleStiffAndForce(domain)

    # @info "Fint: ", Fint, domain.fext

    u = Ksparse\(domain.fext - Fint)
    domain.state[domain.dof_to_eq] .= u

    # Fint, Ksparse = assembleStiffAndForce(domain)
    # @info "Fint, fext: ", Fint -  domain.fext

    return domain
end