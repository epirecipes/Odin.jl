# Composition of epidemiological networks by merging shared species.
#
# Follows the structured cospan pattern: open models expose species as
# interfaces, and composition identifies (merges) shared species.

"""
    compose(nets...; shared=:auto)

Compose multiple `EpiNet`s by merging species with matching names.

When `shared=:auto` (default), species appearing in more than one net are
automatically identified and merged. You can also pass `shared` as a vector
of symbols to control which species are merged.

## Example

```julia
infection = EpiNet([:S => 990.0, :I => 10.0],
                   [:inf => ([:S, :I] => [:I, :I], :beta)])
recovery  = EpiNet([:I => 10.0, :R => 0.0],
                   [:rec => ([:I] => [:R], :gamma)])
sir = compose(infection, recovery)
# Species: [:S, :I, :R], Transitions: [:inf, :rec]
```
"""
function compose(nets::EpiNet...; shared=:auto)
    length(nets) == 0 && error("compose requires at least one network")
    length(nets) == 1 && return deepcopy(nets[1])

    result = EpiNet()
    species_map = Dict{Symbol, Int}()

    # Determine shared species
    if shared === :auto
        all_species = Symbol[]
        for net in nets
            append!(all_species, species_names(net))
        end
        shared_set = Set{Symbol}()
        seen = Set{Symbol}()
        for s in all_species
            if s in seen
                push!(shared_set, s)
            end
            push!(seen, s)
        end
    else
        shared_set = Set{Symbol}(shared)
    end

    for net in nets
        sn = species_names(net)
        sc = species_concentrations(net)

        # Map local species indices to result indices
        local_s_map = Dict{Int, Int}()
        for (local_idx, name) in enumerate(sn)
            if haskey(species_map, name)
                local_s_map[local_idx] = species_map[name]
                # Use max concentration for shared species
                cur = result.net[:concentration][species_map[name]]
                result.net[:concentration][species_map[name]] = max(cur, sc[local_idx])
            else
                new_idx = add_species!(result, name, sc[local_idx])
                species_map[name] = new_idx
                local_s_map[local_idx] = new_idx
            end
        end

        # Copy transitions with remapped arcs
        tn = transition_names(net)
        tr = transition_rates(net)
        for t in 1:ntransitions(net)
            new_t = add_part!(result.net, :T; tname=tn[t], rate=tr[t])
            for a in incident(net.net, t, :it)
                add_part!(result.net, :I; it=new_t, is=local_s_map[net.net[a, :is]])
            end
            for a in incident(net.net, t, :ot)
                add_part!(result.net, :O; ot=new_t, os=local_s_map[net.net[a, :os]])
            end
        end
    end

    return result
end

"""
    compose_with_interface(nets, interface_species)

Compose networks, merging only the explicitly listed interface species.
"""
function compose_with_interface(nets::Vector{EpiNet}, interface::Vector{Symbol})
    compose(nets...; shared=interface)
end
