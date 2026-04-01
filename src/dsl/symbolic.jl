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

function _eval_static_expr(expr, cl::ModelClassification, index_env::Dict{Symbol,Int}=Dict{Symbol,Int}())
    if expr isa Integer
        return Int(expr)
    elseif expr isa AbstractFloat
        return Int(round(expr))
    elseif expr isa Symbol
        if haskey(index_env, expr)
            return index_env[expr]
        elseif haskey(cl.parameters, expr)
            default = cl.parameters[expr].default
            default === nothing && return nothing
            return _eval_static_expr(default, cl, index_env)
        else
            return nothing
        end
    elseif expr isa Expr
        if expr.head == :block
            for arg in expr.args
                arg isa LineNumberNode && continue
                return _eval_static_expr(arg, cl, index_env)
            end
            return nothing
        elseif expr.head == :call
            op = expr.args[1]
            args = [_eval_static_expr(arg, cl, index_env) for arg in expr.args[2:end]]
            any(arg -> arg === nothing, args) && return nothing
            if op == :+
                return reduce(+, args)
            elseif op == :- && length(args) == 1
                return -args[1]
            elseif op == :- && length(args) == 2
                return args[1] - args[2]
            elseif op == :*
                return reduce(*, args)
            elseif op == :/ && length(args) == 2
                return args[1] / args[2]
            elseif op == :^ && length(args) == 2
                return args[1] ^ args[2]
            elseif op == :(==) && length(args) == 2
                return args[1] == args[2]
            elseif op == :(!=) && length(args) == 2
                return args[1] != args[2]
            elseif op == :(<) && length(args) == 2
                return args[1] < args[2]
            elseif op == :(<=) && length(args) == 2
                return args[1] <= args[2]
            elseif op == :(>) && length(args) == 2
                return args[1] > args[2]
            elseif op == :(>=) && length(args) == 2
                return args[1] >= args[2]
            elseif op == :(&&) && length(args) == 2
                return args[1] && args[2]
            elseif op == :(||) && length(args) == 2
                return args[1] || args[2]
            end
        end
    end
    return nothing
end

function _resolve_static_dims(expr, cl::ModelClassification)
    dims = Int[]
    for part in _dim_each(expr)
        val = _eval_static_expr(part, cl)
        val === nothing && return nothing
        push!(dims, Int(val))
    end
    return dims
end

function _linear_index_value(indices::AbstractVector{Int}, dims::AbstractVector{Int})
    length(indices) == 1 && return indices[1]
    idx = indices[1]
    stride = dims[1]
    for k in 2:length(indices)
        idx += (indices[k] - 1) * stride
        k < length(indices) && (stride *= dims[k])
    end
    return idx
end

function _alloc_symbolic_array(prefix::Symbol, dims::AbstractVector{Int})
    n = prod(dims)
    syms = [Symbol(prefix, "_", i) for i in 1:n]
    vars = [Symbolics.variable(sym) for sym in syms]
    arr = length(dims) == 1 ? vars : reshape(copy(vars), dims...)
    return syms, vars, arr
end

function _lhs_index_envs(ex::OdinExpr, cl::ModelClassification, dims::AbstractVector{Int})
    idx_syms, range_bounds = _resolved_lhs_indices(ex, cl)
    isempty(idx_syms) && return [([], Dict{Symbol, Int}())]

    ranges = Vector{UnitRange{Int}}(undef, length(idx_syms))
    for (k, idx) in enumerate(idx_syms)
        if haskey(range_bounds, idx)
            lo, hi = range_bounds[idx]
            lo_val = _eval_static_expr(lo, cl)
            hi_val = _eval_static_expr(hi, cl)
            (lo_val === nothing || hi_val === nothing) && return nothing
            ranges[k] = Int(lo_val):Int(hi_val)
        else
            ranges[k] = 1:dims[min(k, length(dims))]
        end
    end

    envs = Tuple{Vector{Int}, Dict{Symbol, Int}}[]
    for values in Iterators.product(ranges...)
        idx_vals = Int[values...]
        env = Dict{Symbol, Int}(idx_syms[i] => idx_vals[i] for i in eachindex(idx_syms))
        push!(envs, (idx_vals, env))
    end
    return envs
end

"""
    _build_symbolic_rhs(phases, cl, sv_set)

Build symbolic expressions for the ODE right-hand side.
Returns (sym_state, sym_params, sym_time, rhs_exprs, intermediates_dict)
where rhs_exprs[i] is the symbolic expression for dstate[i]/dt.
"""
function _build_symbolic_rhs(phases, cl, sv_set)
    dims = cl.dims
    resolved_dims = Dict{Symbol, Vector{Int}}()
    for (name, dim_expr) in dims
        static_dims = _resolve_static_dims(dim_expr, cl)
        static_dims === nothing && return nothing
        resolved_dims[name] = static_dims
    end

    # Create symbolic variables for flattened state with array views where needed.
    sym_state = Any[]
    state_map = Dict{Symbol, Any}()
    state_binding_stmts = Expr[]
    state_offsets = Dict{Symbol, Int}()
    next_state_idx = 1
    for v in cl.state_vars
        state_offsets[v] = next_state_idx - 1
        if haskey(resolved_dims, v)
            dims_v = resolved_dims[v]
            syms_v, vars_v, arr_v = _alloc_symbolic_array(Symbol("__state_", v), dims_v)
            append!(sym_state, vars_v)
            state_map[v] = arr_v
            for (local_idx, sym) in enumerate(syms_v)
                push!(state_binding_stmts, :($sym = state[$(next_state_idx + local_idx - 1)]))
            end
            next_state_idx += length(vars_v)
        else
            sym = Symbol("__state_", v)
            var = Symbolics.variable(sym)
            push!(sym_state, var)
            state_map[v] = var
            push!(state_binding_stmts, :($sym = state[$next_state_idx]))
            next_state_idx += 1
        end
    end

    # Create symbolic variable for time
    sym_time = Symbolics.variable(:time)

    # Create symbolic variables for all parameters used in the RHS.
    all_params = Symbol[ex.name for ex in phases.create_eqs if ex.type == EXPR_PARAMETER]
    diff_params = _collect_diff_params(cl)
    sym_params = Dict{Symbol, Any}()
    param_binding_stmts = Expr[]
    for p in all_params
        is_array_param = haskey(resolved_dims, p)
        if is_array_param && haskey(cl.parameters, p) && cl.parameters[p].differentiate
            return nothing
        end

        if is_array_param
            dims_p = resolved_dims[p]
            syms_p, _, arr_p = _alloc_symbolic_array(Symbol("__param_", p), dims_p)
            sym_params[p] = arr_p
            for (local_idx, sym) in enumerate(syms_p)
                if length(dims_p) == 1
                    push!(param_binding_stmts, :($sym = pars.$p[$local_idx]))
                else
                    idxs = Tuple(CartesianIndices(Tuple(dims_p))[local_idx])
                    ref = Expr(:ref, :(pars.$p), idxs...)
                    push!(param_binding_stmts, :($sym = $ref))
                end
            end
        else
            sym = Symbol("__param_", p)
            sym_params[p] = Symbolics.variable(sym)
            push!(param_binding_stmts, :($sym = pars.$p))
        end
    end

    sym_diff_params = Any[]
    for p in diff_params
        haskey(sym_params, p) || return nothing
        sym_params[p] isa AbstractArray && return nothing
        push!(sym_diff_params, sym_params[p])
    end

    # Build intermediate expressions
    intermediates = Dict{Symbol, Any}()
    for ex in phases.dynamic_eqs
        ex.type == EXPR_ASSIGNMENT || continue
        if haskey(resolved_dims, ex.name)
            dims_ex = resolved_dims[ex.name]
            envs = _lhs_index_envs(ex, cl, dims_ex)
            envs === nothing && return nothing
            vals = length(dims_ex) == 1 ? Vector{Any}(undef, dims_ex[1]) : Array{Any}(undef, dims_ex...)
            for (idx_vals, env) in envs
                val = _symbolify_expr(ex.rhs, state_map, sym_params, intermediates, sym_time, env, cl)
                val === nothing && return nothing
                vals[idx_vals...] = val
            end
            intermediates[ex.name] = vals
        else
            val = _symbolify_expr(ex.rhs, state_map, sym_params, intermediates, sym_time,
                                  Dict{Symbol, Int}(), cl)
            if val === nothing
                return nothing  # Unsupported expression
            end
            intermediates[ex.name] = val
        end
    end

    # Build derivative expressions
    rhs_exprs = Vector{Any}(undef, length(sym_state))
    for ex in phases.dynamic_eqs
        ex.type == EXPR_DERIV || continue
        if haskey(resolved_dims, ex.name)
            dims_ex = resolved_dims[ex.name]
            envs = _lhs_index_envs(ex, cl, dims_ex)
            envs === nothing && return nothing
            offset = state_offsets[ex.name]
            for (idx_vals, env) in envs
                idx = offset + _linear_index_value(idx_vals, dims_ex)
                val = _symbolify_expr(ex.rhs, state_map, sym_params, intermediates, sym_time, env, cl)
                val === nothing && return nothing
                rhs_exprs[idx] = val
            end
        else
            idx = state_offsets[ex.name] + 1
            val = _symbolify_expr(ex.rhs, state_map, sym_params, intermediates, sym_time,
                                  Dict{Symbol, Int}(), cl)
            if val === nothing
                return nothing
            end
            rhs_exprs[idx] = val
        end
    end

    # Verify all rhs_exprs are assigned
    for i in 1:length(rhs_exprs)
        if !isassigned(rhs_exprs, i)
            return nothing
        end
    end

    return (
        sym_state,
        sym_diff_params,
        diff_params,
        sym_time,
        rhs_exprs,
        state_binding_stmts,
        param_binding_stmts,
    )
end

"""
    _symbolify_expr(expr, state_map, param_map, intermediates, sym_time)

Convert an odin DSL expression into a Symbolics.jl expression.
Returns `nothing` if the expression contains unsupported constructs.
"""
function _symbolify_expr(expr, state_map, param_map, intermediates, sym_time,
                         index_env::Dict{Symbol, Int}, cl::ModelClassification)
    if expr isa Number
        return expr
    elseif expr isa Symbol
        if haskey(index_env, expr)
            return index_env[expr]
        elseif haskey(state_map, expr)
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
            if op == :sum
                if length(expr.args) == 2
                    term = _symbolify_expr(expr.args[2], state_map, param_map, intermediates,
                                           sym_time, index_env, cl)
                    term === nothing && return nothing
                    if term isa AbstractArray
                        return reduce(+, vec(term))
                    else
                        return term
                    end
                elseif length(expr.args) == 3 && expr.args[2] isa Symbol
                    idx = expr.args[2]
                    bound_expr = _find_reduction_bound(expr.args[3], idx, cl)
                    bound_expr === nothing && return nothing
                    bound_val = _eval_static_expr(bound_expr, cl, index_env)
                    bound_val === nothing && return nothing
                    terms = Any[]
                    for k in 1:Int(bound_val)
                        next_env = copy(index_env)
                        next_env[idx] = k
                        val = _symbolify_expr(expr.args[3], state_map, param_map, intermediates,
                                             sym_time, next_env, cl)
                        val === nothing && return nothing
                        push!(terms, val)
                    end
                    return isempty(terms) ? 0 : reduce(+, terms)
                else
                    return nothing
                end
            elseif op == :ifelse && length(expr.args) == 4
                cond = _eval_static_expr(expr.args[2], cl, index_env)
                cond === nothing && return nothing
                branch = cond ? expr.args[3] : expr.args[4]
                return _symbolify_expr(branch, state_map, param_map, intermediates,
                                       sym_time, index_env, cl)
            end

            args = [_symbolify_expr(a, state_map, param_map, intermediates, sym_time, index_env, cl)
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
        elseif expr.head == :ref
            base = expr.args[1]
            haskey(state_map, base) || haskey(intermediates, base) || haskey(param_map, base) || return nothing
            container = haskey(state_map, base) ? state_map[base] :
                        haskey(intermediates, base) ? intermediates[base] :
                        param_map[base]
            indices = Int[]
            for arg in expr.args[2:end]
                arg == :(:) && return nothing
                idx = _eval_static_expr(arg, cl, index_env)
                idx === nothing && return nothing
                push!(indices, Int(idx))
            end
            return container[indices...]
        elseif expr.head == :block
            # Unwrap single-expression blocks
            real_args = filter(a -> !(a isa LineNumberNode), expr.args)
            if length(real_args) == 1
                return _symbolify_expr(real_args[1], state_map, param_map, intermediates,
                                       sym_time, index_env, cl)
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
    # Guard: bail out for large models where symbolic differentiation
    # would take unreasonable time.  The runtime falls back to numerical
    # (ForwardDiff-based) Jacobian when no symbolic version is available.
    n_sv_estimate = length(sv_set)
    if _has_arrays(cl)
        for v in sv_set
            if haskey(cl.dims, v)
                dim_size = _resolve_static_dims(cl.dims[v], cl)
                if dim_size !== nothing
                    n_sv_estimate += prod(dim_size) - 1
                end
            end
        end
    end
    if n_sv_estimate > 15
        return nothing
    end

    result = _build_symbolic_rhs(phases, cl, sv_set)
    if result === nothing
        return nothing
    end

    sym_state, sym_diff_params, diff_param_names, sym_time, rhs_exprs,
        state_binding_stmts, param_binding_stmts = result

    n_state = length(sym_state)
    n_diff = length(diff_param_names)

    if n_diff == 0
        return nothing  # Nothing to differentiate
    end

    # Generate the VJP functions (most useful for adjoint):
    # vjp_state: result[j] = Σ_i J_state[i,j] * v[i]  (= J^T * v)
    # vjp_params: result[jp] = Σ_i J_params[i,jp] * v[i]
    v_sym = [Symbolics.variable(Symbol(:_v_, i)) for i in 1:n_state]
    weighted_rhs = Symbolics.Num(0)
    for i in 1:n_state
        weighted_rhs += rhs_exprs[i] * v_sym[i]
    end

    vjp_state_exprs = Vector{Any}(undef, n_state)
    for j in 1:n_state
        vjp_state_exprs[j] = Symbolics.simplify(Symbolics.derivative(weighted_rhs, sym_state[j]))
    end

    vjp_params_exprs = Vector{Any}(undef, n_diff)
    for jp in 1:n_diff
        vjp_params_exprs[jp] = Symbolics.simplify(Symbolics.derivative(weighted_rhs, sym_diff_params[jp]))
    end

    # Generate Julia code from symbolic expressions using Symbolics.toexpr
    function _sym_to_julia(sym_expr)
        Symbolics.toexpr(sym_expr)
    end

    # Build the VJP state function body
    vjp_state_stmts = Expr[]
    append!(vjp_state_stmts, state_binding_stmts)
    append!(vjp_state_stmts, param_binding_stmts)
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
    append!(vjp_params_stmts, state_binding_stmts)
    append!(vjp_params_stmts, param_binding_stmts)
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

        function Odin._odin_symbolic_n_state(model::$model_name)
            return $(n_state)
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
