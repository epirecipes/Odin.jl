# Monty DSL: a macro for specifying prior distributions that produces a MontyModel.
# Supports scalar and vector parameters with Distributions.jl distributions.
# Provides automatic gradient via ForwardDiff.jl.

"""
    @monty_prior(block)

Define a prior model using distribution syntax. Returns a `MontyModel` with
automatic gradient support.

## Example

```julia
prior = @monty_prior begin
    beta ~ Exponential(1.0)
    gamma ~ Gamma(2.0, 0.1)
end
```

Supports:
- Scalar parameters: `x ~ Distribution(args...)`
- Bounded parameters: constraints from distribution support
- Automatic log-density gradient via ForwardDiff.jl
"""
macro monty_prior(block)
    # Parse the prior block
    specs = _parse_prior_block(block)
    code = _generate_prior_model(specs)
    return esc(code)
end

"""Parsed prior specification for a single parameter."""
struct PriorSpec
    name::Symbol
    dist_expr::Expr      # Distribution constructor call
    src::LineNumberNode
end

function _parse_prior_block(block::Expr)
    specs = PriorSpec[]
    src = LineNumberNode(0, :unknown)

    function walk(e)
        if e isa LineNumberNode
            src = e
        elseif e isa Expr
            if e.head == :block
                for a in e.args; walk(a); end
            elseif e.head == :call && e.args[1] == :(~)
                lhs = e.args[2]
                rhs = e.args[3]
                name = lhs isa Symbol ? lhs : error("Prior LHS must be a symbol, got: $lhs")
                rhs isa Expr && rhs.head == :call ||
                    error("Prior RHS must be a distribution call, got: $rhs")
                push!(specs, PriorSpec(name, rhs, src))
            else
                error("Unsupported expression in @monty_prior: $(e.head)")
            end
        end
    end

    walk(block)
    isempty(specs) && error("@monty_prior block must contain at least one prior specification")
    return specs
end

function _generate_prior_model(specs::Vector{PriorSpec})
    n_pars = length(specs)
    param_names = [string(s.name) for s in specs]

    # Qualify distribution constructors with Odin.Distributions.
    qualified_dists = [_qualify_dist(s.dist_expr) for s in specs]

    # Generate the density function
    density_stmts = Any[]
    push!(density_stmts, :(lp = zero(eltype(x))))
    for (i, qd) in enumerate(qualified_dists)
        push!(density_stmts, :(lp += Odin.Distributions.logpdf($qd, x[$i])))
    end
    push!(density_stmts, :(return lp))

    # Generate domain (from distribution support)
    domain_stmts = Any[]
    push!(domain_stmts, :(_dom = fill(-Inf, $n_pars, 2)))
    push!(domain_stmts, :(_dom[:, 2] .= Inf))
    for (i, qd) in enumerate(qualified_dists)
        push!(domain_stmts, :(let _s = Odin.Distributions.support($qd)
            if _s isa Odin.Distributions.RealInterval
                _dom[$i, 1] = _s.lb
                _dom[$i, 2] = _s.ub
            end
        end))
    end
    push!(domain_stmts, :(_dom))

    quote
        let
            _density = function(x::AbstractVector)
                $(Expr(:block, density_stmts...))
            end

            _gradient = function(x::AbstractVector)
                Odin.ForwardDiff.gradient(_density, x)
            end

            _direct_sample = function(rng::Odin.Random.AbstractRNG)
                x = Vector{Float64}(undef, $n_pars)
                $(Expr(:block, [:(x[$i] = Odin.Random.rand(rng, $(qualified_dists[i])))
                                for i in 1:n_pars]...))
                return x
            end

            _domain = $(Expr(:block, domain_stmts...))

            Odin.monty_model(
                _density;
                parameters=$(QuoteNode(param_names)),
                gradient=_gradient,
                direct_sample=_direct_sample,
                domain=_domain,
                properties=Odin.MontyModelProperties(
                    has_gradient=true,
                    has_direct_sample=true,
                ),
            )
        end
    end
end

"""Qualify distribution constructor with Odin.Distributions. prefix."""
function _qualify_dist(expr::Expr)
    if expr.head == :call
        fname = expr.args[1]
        qualified_name = :(Odin.Distributions.$fname)
        new_args = Any[qualified_name]
        for a in expr.args[2:end]
            push!(new_args, a)
        end
        return Expr(:call, new_args...)
    end
    return expr
end
