# Symbolic differentiation for odin models using Symbolics.jl.
#
# At @odin compile time, we construct symbolic expressions for the ODE RHS,
# differentiate them symbolically w.r.t. state and parameters, and compile
# the results into efficient Julia functions.
#
# This provides:
#   _odin_jacobian_state!  — ∂f/∂y matrix (n_state × n_state)
#   _odin_jacobian_params! — ∂f/∂θ matrix (n_state × n_params)
#   _odin_vjp_state!       — (∂f/∂y)^T * v  (vector-Jacobian product)
#   _odin_vjp_params!      — (∂f/∂θ)^T * v  (vector-Jacobian product)
#
# When symbolic differentiation fails (e.g., complex control flow, external
# calls), we fall back to ReverseDiff.jl at runtime.

using Symbolics

"""
    _collect_diff_params(cl::ModelClassification) -> Vector{Symbol}

Return parameter names marked with `differentiate = true`.
"""
function _collect_diff_params(cl::ModelClassification)
    diff_params = Symbol[]
    for (name, info) in cl.parameters
        if info.differentiate
            push!(diff_params, name)
        end
    end
    return sort(diff_params)
end

"""
    _build_symbolic_rhs(phases, cl, sv_set)

Build symbolic expressions for the ODE right-hand side.
Returns (sym_state, sym_params, sym_time, rhs_exprs, intermediates_dict)
where rhs_exprs[i] is the symbolic expression for dstate[i]/dt.
"""
function _build_symbolic_rhs(phases, cl, sv_set)
    n_state = length(cl.state_vars)
    dims = cl.dims

    # Only handle scalar (non-array) models symbolically for now
    has_arrays = any(v -> haskey(dims, v), cl.state_vars)

    # Create symbolic variables for state
    sym_state = [Symbolics.variable(v) for v in cl.state_vars]
    state_map = Dict{Symbol, Any}(cl.state_vars[i] => sym_state[i] for i in 1:n_state)

    # Create symbolic variable for time
    sym_time = Symbolics.variable(:time)

    # Create symbolic variables for all parameters
    all_params = Symbol[ex.name for ex in phases.create_eqs if ex.type == EXPR_PARAMETER]
    sym_params = Dict{Symbol, Any}(p => Symbolics.variable(p) for p in all_params)

    # Build intermediate expressions
    intermediates = Dict{Symbol, Any}()
    for ex in phases.dynamic_eqs
        ex.type == EXPR_ASSIGNMENT || continue
        if !isempty(ex.indices) && haskey(dims, ex.name)
            # Array intermediate — can't handle symbolically yet
            return nothing
        end
        val = _symbolify_expr(ex.rhs, state_map, sym_params, intermediates, sym_time)
        if val === nothing
            return nothing  # Unsupported expression
        end
        intermediates[ex.name] = val
    end

    # Build derivative expressions
    rhs_exprs = Vector{Any}(undef, n_state)
    for ex in phases.dynamic_eqs
        ex.type == EXPR_DERIV || continue
        if !isempty(ex.indices) && haskey(dims, ex.name)
            return nothing  # Array deriv — can't handle symbolically yet
        end
        idx = findfirst(==(ex.name), cl.state_vars)
        val = _symbolify_expr(ex.rhs, state_map, sym_params, intermediates, sym_time)
        if val === nothing
            return nothing
        end
        rhs_exprs[idx] = val
    end

    # Verify all rhs_exprs are assigned
    for i in 1:n_state
        if !isassigned(rhs_exprs, i)
            return nothing
        end
    end

    diff_params = _collect_diff_params(cl)
    sym_diff_params = [sym_params[p] for p in diff_params]

    return (sym_state, sym_diff_params, diff_params, sym_time, rhs_exprs, all_params, sym_params)
end

"""
    _symbolify_expr(expr, state_map, param_map, intermediates, sym_time)

Convert an odin DSL expression into a Symbolics.jl expression.
Returns `nothing` if the expression contains unsupported constructs.
"""
function _symbolify_expr(expr, state_map, param_map, intermediates, sym_time)
    if expr isa Number
        return expr
    elseif expr isa Symbol
        if haskey(state_map, expr)
            return state_map[expr]
        elseif haskey(intermediates, expr)
            return intermediates[expr]
        elseif haskey(param_map, expr)
            return param_map[expr]
        elseif expr == :time || expr == :t
            return sym_time
        else
            return nothing  # Unknown symbol
        end
    elseif expr isa Expr
        if expr.head == :call
            op = expr.args[1]
            args = [_symbolify_expr(a, state_map, param_map, intermediates, sym_time)
                    for a in expr.args[2:end]]
            if any(a === nothing for a in args)
                return nothing
            end
            # Map common functions — handle n-ary + and *
            if op == :+ && length(args) >= 2
                return reduce(+, args)
            elseif op == :+ && length(args) == 1
                return args[1]
            elseif op == :- && length(args) == 2
                return args[1] - args[2]
            elseif op == :- && length(args) == 1
                return -args[1]
            elseif op == :* && length(args) >= 2
                return reduce(*, args)
            elseif op == :* && length(args) == 1
                return args[1]
            elseif op == :/ && length(args) == 2
                return args[1] / args[2]
            elseif op == :^ && length(args) == 2
                return args[1] ^ args[2]
            elseif op == :sqrt
                return sqrt(args[1])
            elseif op == :exp
                return exp(args[1])
            elseif op == :log
                return log(args[1])
            elseif op == :abs
                return abs(args[1])
            elseif op == :sin
                return sin(args[1])
            elseif op == :cos
                return cos(args[1])
            elseif op == :min && length(args) == 2
                return min(args[1], args[2])
            elseif op == :max && length(args) == 2
                return max(args[1], args[2])
            else
                return nothing  # Unknown function
            end
        elseif expr.head == :block
            # Unwrap single-expression blocks
            real_args = filter(a -> !(a isa LineNumberNode), expr.args)
            if length(real_args) == 1
                return _symbolify_expr(real_args[1], state_map, param_map, intermediates, sym_time)
            end
            return nothing
        end
    end
    return nothing
end

"""
    _gen_symbolic_jacobian(phases, cl, sv_set)

Generate code for symbolic Jacobian methods. Returns a quote block
defining _odin_jacobian_state!, _odin_jacobian_params!, _odin_vjp_state!,
_odin_vjp_params!, and the _odin_has_symbolic_jacobian flag.

Returns `nothing` if symbolic differentiation is not possible for this model.
"""
function _gen_symbolic_jacobian(phases, cl, sv_set, model_name)
    result = _build_symbolic_rhs(phases, cl, sv_set)
    if result === nothing
        return nothing
    end

    sym_state, sym_diff_params, diff_param_names, sym_time, rhs_exprs,
        all_params, sym_params = result

    n_state = length(cl.state_vars)
    n_diff = length(diff_param_names)

    if n_diff == 0
        return nothing  # Nothing to differentiate
    end

    # Compute symbolic Jacobian ∂f/∂y
    J_state = Matrix{Any}(undef, n_state, n_state)
    for i in 1:n_state
        for j in 1:n_state
            d = Symbolics.derivative(rhs_exprs[i], sym_state[j])
            J_state[i, j] = Symbolics.simplify(d)
        end
    end

    # Compute symbolic Jacobian ∂f/∂θ
    J_params = Matrix{Any}(undef, n_state, n_diff)
    for i in 1:n_state
        for jp in 1:n_diff
            d = Symbolics.derivative(rhs_exprs[i], sym_diff_params[jp])
            J_params[i, jp] = Symbolics.simplify(d)
        end
    end

    # Build code that evaluates J_state given state and pars
    # We need to substitute symbolic variables → actual values
    state_syms = cl.state_vars
    all_param_syms = sort(collect(keys(sym_params)))

    # Generate the VJP functions (most useful for adjoint):
    # vjp_state: result[j] = Σ_i J_state[i,j] * v[i]  (= J^T * v)
    # vjp_params: result[jp] = Σ_i J_params[i,jp] * v[i]

    # Compute symbolic VJP expressions
    v_sym = [Symbolics.variable(Symbol(:_v_, i)) for i in 1:n_state]

    vjp_state_exprs = Vector{Any}(undef, n_state)
    for j in 1:n_state
        s = Symbolics.Num(0)
        for i in 1:n_state
            s = s + J_state[i, j] * v_sym[i]
        end
        vjp_state_exprs[j] = Symbolics.simplify(s)
    end

    vjp_params_exprs = Vector{Any}(undef, n_diff)
    for jp in 1:n_diff
        s = Symbolics.Num(0)
        for i in 1:n_state
            s = s + J_params[i, jp] * v_sym[i]
        end
        vjp_params_exprs[jp] = Symbolics.simplify(s)
    end

    # Generate Julia code from symbolic expressions using Symbolics.toexpr
    function _sym_to_julia(sym_expr)
        Symbolics.toexpr(sym_expr)
    end

    # Build the VJP state function body
    vjp_state_stmts = Expr[]
    # Unpack state
    for (i, sv) in enumerate(state_syms)
        push!(vjp_state_stmts, :($sv = state[$i]))
    end
    # Unpack parameters
    for p in all_param_syms
        push!(vjp_state_stmts, :($p = pars.$p))
    end
    # Unpack v vector
    for i in 1:n_state
        vsym = Symbol(:_v_, i)
        push!(vjp_state_stmts, :($vsym = v[$i]))
    end
    # Time
    push!(vjp_state_stmts, :(time = t))
    # Compute VJP
    for j in 1:n_state
        jexpr = _sym_to_julia(vjp_state_exprs[j])
        push!(vjp_state_stmts, :(result[$j] = $jexpr))
    end

    # Build the VJP params function body
    vjp_params_stmts = Expr[]
    for (i, sv) in enumerate(state_syms)
        push!(vjp_params_stmts, :($sv = state[$i]))
    end
    for p in all_param_syms
        push!(vjp_params_stmts, :($p = pars.$p))
    end
    for i in 1:n_state
        vsym = Symbol(:_v_, i)
        push!(vjp_params_stmts, :($vsym = v[$i]))
    end
    push!(vjp_params_stmts, :(time = t))
    for jp in 1:n_diff
        jexpr = _sym_to_julia(vjp_params_exprs[jp])
        push!(vjp_params_stmts, :(result[$jp] = $jexpr))
    end

    vjp_state_body = Expr(:block, vjp_state_stmts...)
    vjp_params_body = Expr(:block, vjp_params_stmts...)

    return quote
        # Flag: this model has symbolic Jacobian
        function Odin._odin_has_symbolic_jacobian(model::$model_name)
            return true
        end

        # Names of differentiated parameters
        function Odin._odin_diff_param_names(model::$model_name)
            return $(QuoteNode(diff_param_names))
        end

        # Number of differentiated parameters
        function Odin._odin_n_diff_params(model::$model_name)
            return $(n_diff)
        end

        # VJP: result = (∂f/∂y)^T * v
        function Odin._odin_vjp_state!(model::$model_name,
                                        result::AbstractVector{T},
                                        state::AbstractVector{T},
                                        v::AbstractVector{T},
                                        pars, t::T) where {T}
            @inbounds $vjp_state_body
            return nothing
        end

        # VJP: result = (∂f/∂θ)^T * v
        function Odin._odin_vjp_params!(model::$model_name,
                                         result::AbstractVector{T},
                                         state::AbstractVector{T},
                                         v::AbstractVector{T},
                                         pars, t::T) where {T}
            @inbounds $vjp_params_body
            return nothing
        end
    end
end
