# EpiNet: high-level wrapper around ACSet-based epidemiological Petri nets.

"""
    EpiNet

An epidemiological network representing a compartmental model as a labelled
Petri net. Species are compartments, transitions are processes (infection,
recovery, etc.), and arcs define stoichiometry.

## Construction

```julia
# From species list and transition specifications
net = EpiNet(
    [:S => 990.0, :I => 10.0, :R => 0.0],
    [:inf => ([:S, :I] => [:I, :I], :beta),
     :rec => ([:I] => [:R], :gamma)]
)

# Using builder API
net = EpiNet()
add_species!(net, :S, 990.0)
add_species!(net, :I, 10.0)
add_species!(net, :R, 0.0)
add_transition!(net, :inf, [:S, :I] => [:I, :I], :beta)
add_transition!(net, :rec, [:I] => [:R], :gamma)
```
"""
struct EpiNet
    net::EpiNetACSet{Symbol, Any, Float64}
end

"""
    EpiNet()

Create an empty epidemiological network.
"""
function EpiNet()
    EpiNet(EpiNetACSet{Symbol, Any, Float64}())
end

"""
    EpiNet(species, transitions)

Create an epidemiological network from species and transition specifications.

- `species`: vector of `name => initial_value` pairs
- `transitions`: vector of `name => (inputs => outputs, rate)` tuples
"""
function EpiNet(species::Vector{<:Pair{Symbol}},
                transitions::Vector{<:Pair{Symbol}})
    net = EpiNet()
    for (name, conc) in species
        add_species!(net, name, Float64(conc))
    end
    for (tname, spec) in transitions
        stoich, rate = spec
        add_transition!(net, tname, stoich, rate)
    end
    return net
end

# ── Species manipulation ─────────────────────────────────────

"""
    add_species!(net, name, concentration=0.0)

Add a compartment to the network. Returns the species index.
"""
function add_species!(net::EpiNet, name::Symbol, conc::Float64=0.0)
    add_part!(net.net, :S; sname=name, concentration=conc)
end

"""
    species_names(net) → Vector{Symbol}

Return the names of all species in the network.
"""
species_names(net::EpiNet) = net.net[:sname]

"""
    species_concentrations(net) → Vector{Float64}

Return the initial concentrations of all species.
"""
species_concentrations(net::EpiNet) = net.net[:concentration]

"""
    nspecies(net) → Int

Number of species (compartments) in the network.
"""
nspecies(net::EpiNet) = nparts(net.net, :S)

# ── Transition manipulation ──────────────────────────────────

"""
    add_transition!(net, name, inputs => outputs, rate)

Add a transition to the network.

- `name`: transition name (Symbol)
- `inputs => outputs`: stoichiometry as `Vector{Symbol} => Vector{Symbol}`
- `rate`: rate expression (Symbol for parameter name, Number, or Expr)

Returns the transition index.
"""
function add_transition!(net::EpiNet, name::Symbol,
                         stoich::Pair{Vector{Symbol}, Vector{Symbol}}, rate)
    inputs, outputs = stoich
    s_names = species_names(net)
    s_lookup = Dict(n => i for (i, n) in enumerate(s_names))
    t_idx = add_part!(net.net, :T; tname=name, rate=rate)
    for s_name in inputs
        s_idx = s_lookup[s_name]
        add_part!(net.net, :I; it=t_idx, is=s_idx)
    end
    for s_name in outputs
        s_idx = s_lookup[s_name]
        add_part!(net.net, :O; ot=t_idx, os=s_idx)
    end
    return t_idx
end

"""
    transition_names(net) → Vector{Symbol}

Return the names of all transitions in the network.
"""
transition_names(net::EpiNet) = net.net[:tname]

"""
    transition_rates(net) → Vector{Any}

Return the rate expressions for all transitions.
"""
transition_rates(net::EpiNet) = net.net[:rate]

"""
    ntransitions(net) → Int

Number of transitions (processes) in the network.
"""
ntransitions(net::EpiNet) = nparts(net.net, :T)

# ── Stoichiometry ────────────────────────────────────────────

"""
    input_species(net, t) → Vector{Symbol}

Return the input species names for transition `t` (by index or name).
"""
function input_species(net::EpiNet, t::Int)
    arcs = incident(net.net, t, :it)
    return [net.net[net.net[a, :is], :sname] for a in arcs]
end
function input_species(net::EpiNet, t::Symbol)
    idx = findfirst(==(t), transition_names(net))
    isnothing(idx) && error("Unknown transition: $t")
    input_species(net, idx)
end

"""
    output_species(net, t) → Vector{Symbol}

Return the output species names for transition `t` (by index or name).
"""
function output_species(net::EpiNet, t::Int)
    arcs = incident(net.net, t, :ot)
    return [net.net[net.net[a, :os], :sname] for a in arcs]
end
function output_species(net::EpiNet, t::Symbol)
    idx = findfirst(==(t), transition_names(net))
    isnothing(idx) && error("Unknown transition: $t")
    output_species(net, idx)
end

"""
    stoichiometry_matrix(net) → Matrix{Int}

Return the net stoichiometry matrix (n_species × n_transitions).
Entry (s, t) is the net change in species s due to transition t.
"""
function stoichiometry_matrix(net::EpiNet)
    ns = nspecies(net)
    nt = ntransitions(net)
    S = zeros(Int, ns, nt)
    for t in 1:nt
        for a in incident(net.net, t, :it)
            s = net.net[a, :is]
            S[s, t] -= 1
        end
        for a in incident(net.net, t, :ot)
            s = net.net[a, :os]
            S[s, t] += 1
        end
    end
    return S
end

"""
    input_matrix(net) → Matrix{Int}

Return the input multiplicity matrix (n_species × n_transitions).
Entry (s, t) is the number of input arcs from species s to transition t.
"""
function input_matrix(net::EpiNet)
    ns = nspecies(net)
    nt = ntransitions(net)
    M = zeros(Int, ns, nt)
    for t in 1:nt
        for a in incident(net.net, t, :it)
            s = net.net[a, :is]
            M[s, t] += 1
        end
    end
    return M
end

# ── Epidemiological building blocks ──────────────────────────

"""
    sir_net(; S0=990.0, I0=10.0, R0=0.0, beta=:beta, gamma=:gamma)

Create a standard SIR model as an EpiNet.
Infection: S + I → 2I (rate β), Recovery: I → R (rate γ).
"""
function sir_net(; S0=990.0, I0=10.0, R0=0.0, beta=:beta, gamma=:gamma)
    EpiNet(
        [:S => S0, :I => I0, :R => R0],
        [:inf => ([:S, :I] => [:I, :I], beta),
         :rec => ([:I] => [:R], gamma)]
    )
end

"""
    seir_net(; S0=990.0, E0=0.0, I0=10.0, R0=0.0,
              beta=:beta, sigma=:sigma, gamma=:gamma)

Create an SEIR model as an EpiNet.
"""
function seir_net(; S0=990.0, E0=0.0, I0=10.0, R0=0.0,
                   beta=:beta, sigma=:sigma, gamma=:gamma)
    EpiNet(
        [:S => S0, :E => E0, :I => I0, :R => R0],
        [:inf => ([:S, :I] => [:E, :I], beta),
         :prog => ([:E] => [:I], sigma),
         :rec  => ([:I] => [:R], gamma)]
    )
end

"""
    sis_net(; S0=990.0, I0=10.0, beta=:beta, gamma=:gamma)

Create an SIS model as an EpiNet.
"""
function sis_net(; S0=990.0, I0=10.0, beta=:beta, gamma=:gamma)
    EpiNet(
        [:S => S0, :I => I0],
        [:inf => ([:S, :I] => [:I, :I], beta),
         :rec => ([:I] => [:S], gamma)]
    )
end

"""
    sirs_net(; S0=990.0, I0=10.0, R0=0.0,
              beta=:beta, gamma=:gamma, delta=:delta)

Create an SIRS model as an EpiNet (recovered lose immunity).
"""
function sirs_net(; S0=990.0, I0=10.0, R0=0.0,
                   beta=:beta, gamma=:gamma, delta=:delta)
    EpiNet(
        [:S => S0, :I => I0, :R => R0],
        [:inf   => ([:S, :I] => [:I, :I], beta),
         :rec   => ([:I] => [:R], gamma),
         :wane  => ([:R] => [:S], delta)]
    )
end

"""
    seirs_net(; S0=990.0, E0=0.0, I0=10.0, R0=0.0,
               beta=:beta, sigma=:sigma, gamma=:gamma, delta=:delta)

Create an SEIRS model as an EpiNet.
"""
function seirs_net(; S0=990.0, E0=0.0, I0=10.0, R0=0.0,
                    beta=:beta, sigma=:sigma, gamma=:gamma, delta=:delta)
    EpiNet(
        [:S => S0, :E => E0, :I => I0, :R => R0],
        [:inf  => ([:S, :I] => [:E, :I], beta),
         :prog => ([:E] => [:I], sigma),
         :rec  => ([:I] => [:R], gamma),
         :wane => ([:R] => [:S], delta)]
    )
end

"""
    sir_vax_net(; S0=990.0, I0=10.0, R0=0.0, V0=0.0,
                 beta=:beta, gamma=:gamma, nu=:nu)

Create an SIR + vaccination model. Vaccination: S → V (rate ν).
V is treated as a removed class (perfect vaccine).
"""
function sir_vax_net(; S0=990.0, I0=10.0, R0=0.0, V0=0.0,
                      beta=:beta, gamma=:gamma, nu=:nu)
    EpiNet(
        [:S => S0, :I => I0, :R => R0, :V => V0],
        [:inf => ([:S, :I] => [:I, :I], beta),
         :rec => ([:I] => [:R], gamma),
         :vax => ([:S] => [:V], nu)]
    )
end

"""
    two_strain_sir_net(; kwargs...)

Create a two-strain SIR model with cross-immunity parameter.
"""
function two_strain_sir_net(; S0=980.0, I1_0=10.0, I2_0=10.0,
                              R1_0=0.0, R2_0=0.0, R12_0=0.0,
                              beta1=:beta1, beta2=:beta2,
                              gamma1=:gamma1, gamma2=:gamma2,
                              sigma=:sigma)
    EpiNet(
        [:S => S0, :I1 => I1_0, :I2 => I2_0,
         :R1 => R1_0, :R2 => R2_0, :R12 => R12_0],
        [:inf1    => ([:S, :I1] => [:I1, :I1], beta1),
         :inf2    => ([:S, :I2] => [:I2, :I2], beta2),
         :rec1    => ([:I1] => [:R1], gamma1),
         :rec2    => ([:I2] => [:R2], gamma2),
         :reinf1  => ([:R2, :I1] => [:I1, :I1], sigma),  # partial cross-immunity
         :reinf2  => ([:R1, :I2] => [:I2, :I2], sigma),
         :rec12a  => ([:I1] => [:R12], gamma1),  # second recovery
         :rec12b  => ([:I2] => [:R12], gamma2)]
    )
end
