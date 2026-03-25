# Lowering: convert epidemiological networks to @odin model code.
#
# Translates a Petri net into odin DSL expressions and compiles them
# through the standard @odin pipeline.

"""
    lower(net; mode=:ode, frequency_dependent=false, N=nothing, params=Dict())

Convert an `EpiNet` to a compiled `DustSystemGenerator`.

## Arguments
- `mode`: `:ode` for deterministic ODE model, `:discrete` for stochastic discrete-time
- `frequency_dependent`: if `true`, divide interaction rates by total population N
- `N`: population size symbol or value (required if `frequency_dependent=true`)
- `params`: `Dict{Symbol,Any}` of default parameter values. Parameters not listed
  default to `parameter()` with no default.

## Returns
A `DustSystemGenerator` ready for use with `dust_system_create`.

## Example

```julia
sir = sir_net()
gen = lower(sir; mode=:ode, frequency_dependent=true, N=:N,
            params=Dict(:beta => 0.3, :gamma => 0.1, :N => 1000.0))
sys = dust_system_create(gen, (beta=0.3, gamma=0.1, N=1000.0))
```
"""
function lower(net::EpiNet; mode::Symbol=:ode, frequency_dependent::Bool=false,
               N=nothing, params::Dict{Symbol,<:Any}=Dict{Symbol,Any}())
    body = lower_expr(net; mode=mode, frequency_dependent=frequency_dependent,
                      N=N, params=params)
    block = Expr(:block, body...)
    _compile_odin_block(block)
end

"""
    lower_expr(net; mode=:ode, frequency_dependent=false, N=nothing, params=Dict())

Generate odin DSL expressions from an `EpiNet` (for inspection/modification).

Returns a `Vector{Expr}` of odin-compatible statements.
"""
function lower_expr(net::EpiNet; mode::Symbol=:ode, frequency_dependent::Bool=false,
                    N=nothing, params::Dict{Symbol,<:Any}=Dict{Symbol,Any}())
    if frequency_dependent && isnothing(N)
        error("N must be specified when frequency_dependent=true")
    end

    s_names = species_names(net)
    s_concs = species_concentrations(net)
    t_names = transition_names(net)
    t_rates = transition_rates(net)
    ns = nspecies(net)
    nt = ntransitions(net)

    stmts = Expr[]

    # Collect all parameter symbols referenced in rates
    rate_params = Set{Symbol}()
    for r in t_rates
        _collect_rate_params!(rate_params, r)
    end
    if frequency_dependent && N isa Symbol
        push!(rate_params, N)
    end
    # Add params from the user dict that aren't rate params
    for k in keys(params)
        push!(rate_params, k)
    end

    # Parameter declarations
    for p in sort(Base.collect(rate_params))
        if haskey(params, p)
            push!(stmts, :($(p) = parameter($(params[p]))))
        else
            push!(stmts, :($(p) = parameter()))
        end
    end

    if mode == :discrete
        # dt is provided by the system, not declared as a parameter
    end

    # Initial conditions
    for s in 1:ns
        push!(stmts, :(initial($(s_names[s])) = $(s_concs[s])))
    end

    if mode == :ode
        _lower_ode!(stmts, net, s_names, t_names, t_rates,
                    ns, nt, frequency_dependent, N)
    elseif mode == :discrete
        _lower_discrete!(stmts, net, s_names, t_names, t_rates,
                         ns, nt, frequency_dependent, N)
    else
        error("Unknown mode: $mode. Use :ode or :discrete")
    end

    return stmts
end

# ── ODE lowering ─────────────────────────────────────────────

function _lower_ode!(stmts, net, s_names, t_names, t_rates,
                     ns, nt, freq_dep, N)
    # Compute propensity for each transition
    for t in 1:nt
        in_arcs = incident(net.net, t, :it)
        prop = _build_propensity(net, t, in_arcs, t_rates[t], freq_dep, N)
        flow_name = Symbol(:flow_, t_names[t])
        push!(stmts, :($flow_name = $prop))
    end

    # Compute derivatives from stoichiometry
    stoich = stoichiometry_matrix(net)
    for s in 1:ns
        terms = Any[]
        for t in 1:nt
            c = stoich[s, t]
            c == 0 && continue
            flow_name = Symbol(:flow_, t_names[t])
            if c == 1
                push!(terms, :($flow_name))
            elseif c == -1
                push!(terms, :(- $flow_name))
            elseif c > 0
                push!(terms, :($c * $flow_name))
            else
                push!(terms, :($(c) * $flow_name))
            end
        end
        if isempty(terms)
            rhs = 0.0
        elseif length(terms) == 1
            rhs = terms[1]
        else
            # Build addition expression
            rhs = terms[1]
            for i in 2:length(terms)
                t_expr = terms[i]
                if t_expr isa Expr && t_expr.head == :call && t_expr.args[1] == :- && length(t_expr.args) == 2
                    rhs = Expr(:call, :-, rhs, t_expr.args[2])
                else
                    rhs = Expr(:call, :+, rhs, t_expr)
                end
            end
        end
        push!(stmts, :(deriv($(s_names[s])) = $rhs))
    end
end

# ── Discrete/stochastic lowering ─────────────────────────────

function _lower_discrete!(stmts, net, s_names, t_names, t_rates,
                          ns, nt, freq_dep, N)
    # For each transition, compute hazard rate and draw events
    # Group transitions by their consumed species to handle competing hazards

    # Use NET stoichiometry to determine which species are truly consumed
    # (net < 0) vs catalytic (net ≥ 0) for each transition.
    stoich = stoichiometry_matrix(net)

    # Build map: species → transitions that NET-consume it (stoich < 0)
    species_consumers = Dict{Int, Vector{Int}}()
    for t in 1:nt
        for s in 1:ns
            if stoich[s, t] < 0
                push!(get!(species_consumers, s, Int[]), t)
            end
        end
    end

    # For each transition, build per-capita hazard rate.
    # Hazard = rate × product(non-consumed input species).
    # The consumed species provides the Binomial count.
    for t in 1:nt
        in_arcs = incident(net.net, t, :it)
        hz = _build_hazard_rate(net, t, in_arcs, t_rates[t], freq_dep, N)
        hz_name = Symbol(:hz_, t_names[t])
        push!(stmts, :($hz_name = $hz))
    end

    event_vars = Dict{Int, Symbol}()

    for s in 1:ns
        consumers = get(species_consumers, s, Int[])
        isempty(consumers) && continue

        if length(consumers) == 1
            t = consumers[1]
            hz_name = Symbol(:hz_, t_names[t])
            prob_name = Symbol(:p_, t_names[t])
            event_name = Symbol(:n_, t_names[t])
            mult = abs(stoich[s, t])
            push!(stmts, :($prob_name = 1.0 - exp(-$hz_name * dt)))
            push!(stmts, :($event_name = Binomial($(s_names[s]), $prob_name)))
            event_vars[t] = event_name
        else
            # Competing hazards: draw total then allocate proportionally
            hz_names = [Symbol(:hz_, t_names[t]) for t in consumers]
            total_hz = Symbol(:hz_total_, s_names[s])
            total_prob = Symbol(:p_total_, s_names[s])
            total_events = Symbol(:n_total_, s_names[s])

            total_expr = Expr(:call, :+, hz_names...)
            push!(stmts, :($total_hz = $total_expr))
            push!(stmts, :($total_prob = 1.0 - exp(-$total_hz * dt)))
            push!(stmts, :($total_events = Binomial($(s_names[s]), $total_prob)))

            # Sequential conditional binomials for allocation
            remaining = total_events
            for (i, t) in enumerate(consumers)
                event_name = Symbol(:n_, t_names[t])
                if i < length(consumers)
                    hz_name = hz_names[i]
                    cond_prob = Symbol(:cp_, t_names[t])
                    push!(stmts, :($cond_prob = $hz_name / ($total_hz + 1e-30)))
                    push!(stmts, :($event_name = Binomial($remaining, $cond_prob)))
                    new_remaining = Symbol(:rem_, s_names[s], :_, i)
                    push!(stmts, :($new_remaining = $remaining - $event_name))
                    remaining = new_remaining
                else
                    push!(stmts, :($event_name = $remaining))
                end
                event_vars[t] = event_name
            end
        end
    end

    # Build update equations from stoichiometry
    for s in 1:ns
        terms = Any[s_names[s]]
        for t in 1:nt
            c = stoich[s, t]
            c == 0 && continue
            haskey(event_vars, t) || continue
            ev = event_vars[t]
            if c == 1
                push!(terms, ev)
            elseif c == -1
                push!(terms, :(- $ev))
            elseif c > 0
                push!(terms, :($c * $ev))
            else
                push!(terms, :($(c) * $ev))
            end
        end
        if length(terms) == 1
            rhs = terms[1]
        else
            rhs = terms[1]
            for i in 2:length(terms)
                t_expr = terms[i]
                if t_expr isa Expr && t_expr.head == :call && t_expr.args[1] == :- && length(t_expr.args) == 2
                    rhs = Expr(:call, :-, rhs, t_expr.args[2])
                else
                    rhs = Expr(:call, :+, rhs, t_expr)
                end
            end
        end
        push!(stmts, :(update($(s_names[s])) = $rhs))
    end
end

# ── Helper functions ─────────────────────────────────────────

"""Build mass-action propensity expression for a transition."""
function _build_propensity(net, t, in_arcs, rate_expr, freq_dep, N)
    # Mass action: rate × Π(input species)
    in_species = Symbol[]
    for a in in_arcs
        push!(in_species, species_names(net)[net.net[a, :is]])
    end

    if isempty(in_species)
        return rate_expr
    end

    # Build product of input species
    prop = rate_expr
    for s in in_species
        prop = :($prop * $s)
    end

    # Frequency-dependent: divide by N^(n_inputs - 1)
    if freq_dep && length(in_species) > 1
        n_extra = length(in_species) - 1
        if n_extra == 1
            prop = :($prop / $N)
        else
            prop = :($prop / $N ^ $n_extra)
        end
    end

    return prop
end

"""Build per-capita hazard rate for stochastic transitions."""
function _build_hazard_rate(net, t, in_arcs, rate_expr, freq_dep, N)
    # For a transition consuming species S (and possibly interacting with I):
    # hazard = rate × Π(other input species)
    # (the species itself provides the "count" via Binomial)
    in_species = Symbol[]
    for a in in_arcs
        push!(in_species, species_names(net)[net.net[a, :is]])
    end

    if length(in_species) <= 1
        return rate_expr
    end

    # The first input species is the one being "consumed" (provides the count).
    # Build hazard from rate × product of remaining species.
    # For infection S + I → 2I: hazard for S is rate × I
    other_species = in_species[2:end]
    hz = rate_expr
    for s in other_species
        hz = :($hz * $s)
    end

    if freq_dep
        n_others = length(other_species)
        if n_others >= 1
            hz = :($hz / $N)
        end
    end

    return hz
end

"""Collect parameter symbols from a rate expression."""
function _collect_rate_params!(params::Set{Symbol}, expr)
    if expr isa Symbol
        push!(params, expr)
    elseif expr isa Expr
        for a in expr.args
            a isa Symbol && a != :* && a != :+ && a != :- && a != :/ && a != :^ &&
                push!(params, a)
            a isa Expr && _collect_rate_params!(params, a)
        end
    end
    # Numbers are ignored
end

"""Compile an odin block expression through the standard pipeline."""
function _compile_odin_block(block::Expr)
    exprs = parse_odin_block(block)
    classification = classify_variables(exprs)
    dep_entries = build_dependency_graph(exprs, classification)
    phases = organise_phases(exprs, classification, dep_entries)
    code = generate_system(exprs, classification, phases)
    return Core.eval(Odin, code)
end
