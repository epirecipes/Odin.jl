# The @odin_model macro: convenience wrapper combining @odin, @monty_prior, and monty_packer.

"""
    @odin_model(block)

Define an odin model with priors and fixed parameters in a single block.
Returns a named tuple `(system=..., prior=..., packer=...)`.

The block may contain:
- Standard odin DSL expressions (deriv, initial, parameter, compare, etc.)
- `@prior begin ... end` — DynamicPPL-style prior specifications
- `@fixed name1 = val1 name2 = val2 ...` — fixed parameter values

Parameters listed in `@prior` become free (sampled) parameters. Parameters
listed in `@fixed` become fixed values in the packer. All other `parameter()`
declarations are included in the odin system but do not appear in the packer.

## Example

```julia
model = @odin_model begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0

    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10.0)
    N = parameter(1000.0)

    @prior begin
        beta ~ Gamma(2.0, 0.25)
        gamma ~ Gamma(2.0, 0.05)
    end

    @fixed I0 = 10.0 N = 1000.0
end

# model.system — DustSystemGenerator
# model.prior  — MontyModel (from @monty_prior)
# model.packer — MontyPacker with free params [:beta, :gamma] and fixed (I0, N)
```
"""
macro odin_model(block)
    odin_block, prior_block, fixed_pairs = _split_odin_model_block(block)

    prior_specs = _parse_prior_block(prior_block)
    free_names = [s.name for s in prior_specs]

    prior_code = _generate_prior_model(prior_specs)

    fixed_nt_expr = if isempty(fixed_pairs)
        :(NamedTuple())
    else
        # Build (key1 = val1, key2 = val2, ...) NamedTuple literal
        pairs = [Expr(:(=), k, v) for (k, v) in fixed_pairs]
        Expr(:tuple, pairs...)
    end

    free_syms_expr = Expr(:vect, [QuoteNode(n) for n in free_names]...)

    # Re-parse the odin block inline so @odin's pipeline runs at compile time.
    # We call the same internal functions that @odin uses.
    odin_exprs = parse_odin_block(odin_block)
    classification = classify_variables(odin_exprs)
    dep_entries = build_dependency_graph(odin_exprs, classification)
    phases = organise_phases(odin_exprs, classification, dep_entries)
    system_code = generate_system(odin_exprs, classification, phases)

    return esc(quote
        let
            _system = $system_code
            _prior = $prior_code
            _packer = Odin.monty_packer($free_syms_expr; fixed=$fixed_nt_expr)
            (system = _system, prior = _prior, packer = _packer)
        end
    end)
end

"""
    _split_odin_model_block(block) -> (odin_block, prior_block, fixed_pairs)

Split an `@odin_model` block into:
- `odin_block` — an Expr(:block, ...) with only odin DSL expressions
- `prior_block` — an Expr(:block, ...) with prior `~` statements
- `fixed_pairs` — Vector{Pair{Symbol,Any}} of fixed parameter assignments
"""
function _split_odin_model_block(block::Expr)
    block.head == :block || error("@odin_model expects a begin...end block")

    odin_args = Any[]
    prior_block = Expr(:block)
    fixed_pairs = Pair{Symbol,Any}[]

    i = 1
    while i <= length(block.args)
        ex = block.args[i]

        if ex isa LineNumberNode
            push!(odin_args, ex)
            i += 1
        elseif _is_macrocall(ex, :prior)
            _extract_prior!(prior_block, ex)
            i += 1
        elseif _is_macrocall(ex, :fixed)
            _extract_fixed!(fixed_pairs, ex)
            i += 1
        else
            push!(odin_args, ex)
            i += 1
        end
    end

    isempty(prior_block.args) && error("@odin_model block must contain a @prior section")

    return Expr(:block, odin_args...), prior_block, fixed_pairs
end

"""Check if an expression is a macrocall to `@name`."""
function _is_macrocall(ex, name::Symbol)
    ex isa Expr || return false
    ex.head == :macrocall || return false
    macro_name = ex.args[1]
    macro_name isa Symbol && return macro_name == Symbol("@", name)
    macro_name isa GlobalRef && return macro_name.name == Symbol("@", name)
    return false
end

"""Extract prior specifications from a `@prior begin ... end` macrocall."""
function _extract_prior!(prior_block::Expr, ex::Expr)
    # macrocall args: [macro_name, line_number_node, body...]
    for a in ex.args[2:end]
        a isa LineNumberNode && continue
        if a isa Expr && a.head == :block
            append!(prior_block.args, a.args)
        else
            push!(prior_block.args, a)
        end
    end
end

"""Extract fixed parameter assignments from `@fixed name=val ...`."""
function _extract_fixed!(pairs::Vector{Pair{Symbol,Any}}, ex::Expr)
    for a in ex.args[2:end]
        a isa LineNumberNode && continue
        if a isa Expr && a.head == :(=)
            lhs = a.args[1]
            rhs = a.args[2]
            lhs isa Symbol || error("@fixed: LHS must be a symbol, got $lhs")
            push!(pairs, lhs => rhs)
        elseif !(a isa LineNumberNode)
            error("@fixed expects assignments (name = value), got: $a")
        end
    end
end
