# Stratification: replicate epidemiological networks across groups.
#
# Given a base model (e.g. SIR) and a set of groups (e.g. age classes),
# produce a stratified model where each group has its own copy of each
# compartment, and interaction transitions use a contact matrix.

"""
    stratify(net, groups; contact=nothing, interaction_transitions=nothing)

Stratify an `EpiNet` across groups (e.g. age classes, spatial patches).

Each species `X` is replicated as `X_group` for each group. Each transition
is replicated per group. Transitions that involve interactions between
species (e.g. infection: S + I → 2I) use the `contact` matrix to create
inter-group transitions.

## Arguments
- `net`: base `EpiNet` to stratify
- `groups`: vector of group labels (e.g. `[:young, :old]`)
- `contact`: square matrix of contact rates between groups (optional).
  If `nothing`, groups interact only within themselves (identity matrix).
- `interaction_transitions`: which transitions involve inter-group mixing.
  These are transitions with ≥2 distinct input species (auto-detected if `nothing`).

## Example

```julia
sir = sir_net()
C = [2.0 0.5; 0.5 1.0]  # contact matrix
sir_age = stratify(sir, [:young, :old]; contact=C)
```
"""
function stratify(net::EpiNet, groups::Vector{Symbol};
                  contact::Union{Nothing, AbstractMatrix{<:Real}}=nothing,
                  interaction_transitions::Union{Nothing, Vector{Symbol}}=nothing)
    ng = length(groups)
    ns = nspecies(net)
    nt = ntransitions(net)

    if !isnothing(contact)
        size(contact) == (ng, ng) || error(
            "Contact matrix must be $(ng)×$(ng), got $(size(contact))")
    end

    # Auto-detect interaction transitions (those with ≥2 distinct input species)
    if isnothing(interaction_transitions)
        interaction_set = Set{Int}()
        for t in 1:nt
            in_arcs = incident(net.net, t, :it)
            in_species = Set(net.net[a, :is] for a in in_arcs)
            if length(in_species) >= 2
                push!(interaction_set, t)
            end
        end
    else
        tnames = transition_names(net)
        interaction_set = Set{Int}()
        for name in interaction_transitions
            idx = findfirst(==(name), tnames)
            isnothing(idx) && error("Unknown transition: $name")
            push!(interaction_set, idx)
        end
    end

    result = EpiNet()
    s_names = species_names(net)
    s_concs = species_concentrations(net)
    t_names = transition_names(net)
    t_rates = transition_rates(net)

    # Create stratified species: X → X_group for each group
    # species_idx_map[local_s, group_idx] → result species index
    species_idx_map = Matrix{Int}(undef, ns, ng)
    for (gi, group) in enumerate(groups)
        for s in 1:ns
            name = Symbol(s_names[s], :_, group)
            idx = add_species!(result, name, s_concs[s] / ng)
            species_idx_map[s, gi] = idx
        end
    end

    for t in 1:nt
        in_arcs = incident(net.net, t, :it)
        out_arcs = incident(net.net, t, :ot)

        if t in interaction_set && !isnothing(contact)
            _add_interaction_transitions!(result, net, t, in_arcs, out_arcs,
                                         groups, species_idx_map, contact,
                                         t_names[t], t_rates[t])
        else
            # Simple within-group replication
            for (gi, group) in enumerate(groups)
                tname = Symbol(t_names[t], :_, group)
                new_t = add_part!(result.net, :T; tname=tname, rate=t_rates[t])
                for a in in_arcs
                    add_part!(result.net, :I; it=new_t,
                              is=species_idx_map[net.net[a, :is], gi])
                end
                for a in out_arcs
                    add_part!(result.net, :O; ot=new_t,
                              os=species_idx_map[net.net[a, :os], gi])
                end
            end
        end
    end

    return result
end

"""Add interaction transitions with contact matrix mixing."""
function _add_interaction_transitions!(result, net, t, in_arcs, out_arcs,
                                       groups, species_idx_map, contact,
                                       base_tname, base_rate)
    ng = length(groups)

    # Identify the "catalyst" (infectious) species — the input species that
    # also appears in outputs (e.g., I in S+I→2I)
    in_species = [net.net[a, :is] for a in in_arcs]
    out_species = [net.net[a, :os] for a in out_arcs]
    catalyst_species = Base.intersect(Set(in_species), Set(out_species))

    # Count how many times each catalyst appears in input — those stay in gj.
    # Extra output copies are "product" (new infections) → go to gi.
    catalyst_in_count = Dict{Int, Int}()
    for s in in_species
        if s in catalyst_species
            catalyst_in_count[s] = get(catalyst_in_count, s, 0) + 1
        end
    end

    for gi in 1:ng
        for gj in 1:ng
            c = contact[gi, gj]
            iszero(c) && continue

            tname = Symbol(base_tname, :_, groups[gi], :_, groups[gj])
            # Scale the rate by the contact matrix entry
            if c == 1.0
                rate = base_rate
            else
                rate = :($c * $base_rate)
            end
            new_t = add_part!(result.net, :T; tname=tname, rate=rate)

            # Input arcs: catalyst species come from group gj (source of infection),
            # susceptible species come from group gi
            for a in in_arcs
                s = net.net[a, :is]
                group_idx = (s in catalyst_species) ? gj : gi
                add_part!(result.net, :I; it=new_t,
                          is=species_idx_map[s, group_idx])
            end

            # Output arcs: for catalyst species, only the first N copies
            # (matching input count) stay in gj; extras go to gi (new product).
            # Non-catalyst outputs go to gi.
            catalyst_remaining = copy(catalyst_in_count)
            for a in out_arcs
                s = net.net[a, :os]
                if s in catalyst_species
                    rem = get(catalyst_remaining, s, 0)
                    if rem > 0
                        # Catalytic copy — stays in source group gj
                        group_idx = gj
                        catalyst_remaining[s] = rem - 1
                    else
                        # Product copy — new infection goes to target group gi
                        group_idx = gi
                    end
                else
                    group_idx = gi
                end
                add_part!(result.net, :O; ot=new_t,
                          os=species_idx_map[s, group_idx])
            end
        end
    end
end
