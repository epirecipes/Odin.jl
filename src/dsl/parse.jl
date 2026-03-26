# Expression parsing for the odin DSL.
# Converts raw Julia Expr into structured OdinExpr representations.

"""Classification of an odin expression."""
@enum ExprType begin
    EXPR_DERIV          # deriv(X) = ...
    EXPR_UPDATE         # update(X) = ...
    EXPR_INITIAL        # initial(X) = ...
    EXPR_DIFFUSION      # diffusion(X) = ...
    EXPR_DIM            # dim(X) = ...
    EXPR_OUTPUT         # output(X) = ...
    EXPR_PARAMETER      # X = parameter(...)
    EXPR_DATA           # X = data()
    EXPR_INTERPOLATE    # X = interpolate(...)
    EXPR_ASSIGNMENT     # X = <expr>
    EXPR_COMPARE        # X ~ Distribution(...)
end

"""Parsed representation of a single odin equation."""
struct OdinExpr
    type::ExprType
    name::Symbol
    rhs::Any                    # the RHS expression or parsed metadata
    indices::Vector{Symbol}     # LHS array indices (empty for scalars)
    src::LineNumberNode         # source location
    range_bounds::Dict{Symbol, Tuple{Any, Any}}  # index => (lo, hi) for range-based indexing
end

# Convenience constructors (backward compatible)
OdinExpr(type, name, rhs, indices, src) = OdinExpr(type, name, rhs, indices, src, Dict{Symbol, Tuple{Any, Any}}())

"""Parsed parameter metadata."""
struct ParameterInfo
    default::Any                # default value or nothing
    type::Symbol                # :real, :integer, :logical
    constant::Bool
    differentiate::Bool
    rank::Int                   # 0 for scalar, >0 for array
    min::Any                    # nothing or numeric
    max::Any                    # nothing or numeric
end

"""Parsed interpolation metadata."""
struct InterpolateInfo
    time_var::Symbol
    value_var::Symbol
    mode::Symbol                # :constant, :linear, :spline
end

"""Parsed comparison (data likelihood) metadata."""
struct CompareInfo
    distribution::Symbol        # e.g., :Normal, :Poisson
    args::Vector{Any}           # distribution arguments as expressions
end

"""
    parse_odin_expr(expr, src)

Parse a single odin DSL expression into an `OdinExpr`.
"""
function parse_odin_expr(expr::Expr, src::LineNumberNode)
    if expr.head == :(=) || expr.head == :(:=)
        return _parse_assignment(expr, src)
    elseif expr.head == :call && expr.args[1] == :(~)
        return _parse_compare(expr, src)
    elseif expr.head == :macrocall
        error("Macro calls are not supported inside @odin blocks")
    else
        error("Unsupported expression type: $(expr.head) in $expr")
    end
end

function _parse_assignment(expr::Expr, src::LineNumberNode)
    lhs = expr.args[1]
    rhs_expr = expr.args[2]

    # LHS is a function call: initial(X), update(X), deriv(X), dim(X), output(X)
    if lhs isa Expr && lhs.head == :call
        fname = lhs.args[1]
        if fname in SPECIAL_LHS
            return _parse_special_lhs(fname, lhs, rhs_expr, src)
        else
            error("Unknown special function on LHS: $fname")
        end
    # LHS is an indexed expression: X[i] or X[i,j]
    elseif lhs isa Expr && lhs.head == :ref
        return _parse_indexed_assignment(lhs, rhs_expr, src)
    # LHS is a plain symbol
    elseif lhs isa Symbol
        return _parse_plain_assignment(lhs, rhs_expr, src)
    else
        error("Invalid LHS: $lhs")
    end
end

function _parse_special_lhs(fname::Symbol, lhs::Expr, rhs_expr, src::LineNumberNode)
    # Extract variable name and optional indices
    inner_args = lhs.args[2:end]
    if length(inner_args) == 0
        error("$fname() requires a variable name")
    end

    # Handle shared dim: dim(S, E, I, R) = n → multiple EXPR_DIM entries
    if fname == :dim && length(inner_args) > 1 && all(a isa Symbol for a in inner_args)
        return [OdinExpr(EXPR_DIM, v, rhs_expr, Symbol[], src, Dict{Symbol, Tuple{Any, Any}}())
                for v in inner_args]
    end

    first_arg = inner_args[1]
    varname, indices, kwargs, range_bounds = _extract_lhs_target(first_arg, inner_args)

    if fname == :initial
        zero_every = get(kwargs, :zero_every, nothing)
        return OdinExpr(EXPR_INITIAL, varname, (rhs=rhs_expr, zero_every=zero_every), indices, src, range_bounds)
    elseif fname == :update
        return OdinExpr(EXPR_UPDATE, varname, rhs_expr, indices, src, range_bounds)
    elseif fname == :deriv
        return OdinExpr(EXPR_DERIV, varname, rhs_expr, indices, src, range_bounds)
    elseif fname == :diffusion
        return OdinExpr(EXPR_DIFFUSION, varname, rhs_expr, indices, src, range_bounds)
    elseif fname == :dim
        return OdinExpr(EXPR_DIM, varname, rhs_expr, indices, src, range_bounds)
    elseif fname == :output
        return OdinExpr(EXPR_OUTPUT, varname, rhs_expr, indices, src, range_bounds)
    else
        error("Unhandled special LHS: $fname")
    end
end

function _extract_lhs_target(first_arg, all_args)
    kwargs = Dict{Symbol, Any}()
    indices = Symbol[]
    range_bounds = Dict{Symbol, Tuple{Any, Any}}()

    # Handle keyword arguments (e.g., initial(x, zero_every=1))
    for arg in all_args[2:end]
        if arg isa Expr && arg.head == :kw
            kwargs[arg.args[1]] = arg.args[2]
        end
    end

    # Standard implicit index variable names by dimension
    _index_names = [:i, :j, :k, :l, :m, :n, :p, :q]

    if first_arg isa Symbol
        return first_arg, indices, kwargs, range_bounds
    elseif first_arg isa Expr && first_arg.head == :ref
        varname = first_arg.args[1]
        for (dim_pos, a) in enumerate(first_arg.args[2:end])
            if a isa Symbol
                push!(indices, a)
            elseif a isa Integer
                # Fixed integer index like S[i, 1] — create a synthetic loop var pinned to that value
                idx_var = dim_pos <= length(_index_names) ? _index_names[dim_pos] : Symbol("idx_$dim_pos")
                push!(indices, idx_var)
                range_bounds[idx_var] = (a, a)
            elseif a isa Expr && a.head == :call && a.args[1] == :(:)
                # Range expression like 2:k_E — infer loop variable
                lo = a.args[2]
                hi = a.args[3]
                idx_var = dim_pos <= length(_index_names) ? _index_names[dim_pos] : Symbol("idx_$dim_pos")
                push!(indices, idx_var)
                range_bounds[idx_var] = (lo, hi)
            end
        end
        return varname, indices, kwargs, range_bounds
    else
        error("Invalid target in special LHS: $first_arg")
    end
end

function _parse_indexed_assignment(lhs::Expr, rhs_expr, src::LineNumberNode)
    varname = lhs.args[1]
    indices = Symbol[]
    for a in lhs.args[2:end]
        if a isa Symbol
            push!(indices, a)
        end
    end

    # Check if RHS is a special function
    if rhs_expr isa Expr && rhs_expr.head == :call
        rhs_fname = rhs_expr.args[1]
        if rhs_fname == :parameter
            pinfo = _parse_parameter_call(rhs_expr)
            return OdinExpr(EXPR_PARAMETER, varname, pinfo, indices, src)
        elseif rhs_fname == :data
            return OdinExpr(EXPR_DATA, varname, nothing, indices, src)
        end
    end

    return OdinExpr(EXPR_ASSIGNMENT, varname, rhs_expr, indices, src)
end

function _parse_plain_assignment(varname::Symbol, rhs_expr, src::LineNumberNode)
    if varname in RESERVED_NAMES
        error("Cannot assign to reserved name: $varname")
    end

    if rhs_expr isa Expr && rhs_expr.head == :call
        rhs_fname = rhs_expr.args[1]
        if rhs_fname == :parameter
            pinfo = _parse_parameter_call(rhs_expr)
            return OdinExpr(EXPR_PARAMETER, varname, pinfo, Symbol[], src)
        elseif rhs_fname == :data
            return OdinExpr(EXPR_DATA, varname, nothing, Symbol[], src)
        elseif rhs_fname == :interpolate
            iinfo = _parse_interpolate_call(rhs_expr)
            return OdinExpr(EXPR_INTERPOLATE, varname, iinfo, Symbol[], src)
        end
    end

    return OdinExpr(EXPR_ASSIGNMENT, varname, rhs_expr, Symbol[], src)
end

function _parse_compare(expr::Expr, src::LineNumberNode)
    # X ~ Distribution(args...)
    lhs = expr.args[2]
    rhs = expr.args[3]

    varname = lhs isa Symbol ? lhs : (lhs isa Expr && lhs.head == :ref ? lhs.args[1] : error("Invalid LHS in comparison: $lhs"))
    indices = Symbol[]
    if lhs isa Expr && lhs.head == :ref
        indices = Symbol[a for a in lhs.args[2:end] if a isa Symbol]
    end

    if !(rhs isa Expr && rhs.head == :call)
        error("RHS of ~ must be a distribution call, got: $rhs")
    end

    dist_name = rhs.args[1]
    if !haskey(DISTRIBUTION_MAP, dist_name)
        error("Unknown distribution: $dist_name")
    end

    dist_args = rhs.args[2:end]
    cinfo = CompareInfo(dist_name, collect(Any, dist_args))
    return OdinExpr(EXPR_COMPARE, varname, cinfo, indices, src)
end

function _parse_parameter_call(expr::Expr)
    args = expr.args[2:end]
    default = nothing
    type = :real
    constant = false
    differentiate = false
    rank = 0
    pmin = nothing
    pmax = nothing

    for (i, arg) in enumerate(args)
        if arg isa Expr && arg.head == :kw
            kw_name = arg.args[1]
            kw_val = arg.args[2]
            if kw_name == :type
                type = kw_val isa QuoteNode ? kw_val.value : kw_val
            elseif kw_name == :constant
                constant = kw_val
            elseif kw_name == :differentiate
                differentiate = kw_val
            elseif kw_name == :rank
                rank = kw_val
            elseif kw_name == :min
                pmin = kw_val
            elseif kw_name == :max
                pmax = kw_val
            end
        elseif i == 1 && !(arg isa Expr && arg.head == :kw)
            default = arg
        end
    end

    return ParameterInfo(default, type, constant, differentiate, rank, pmin, pmax)
end

function _parse_interpolate_call(expr::Expr)
    args = expr.args[2:end]
    length(args) >= 2 || error("interpolate() requires at least 2 arguments")
    time_var = args[1]
    value_var = args[2]
    mode = length(args) >= 3 ? (args[3] isa QuoteNode ? args[3].value : Symbol(args[3])) : :linear

    time_var isa Symbol || error("interpolate() first argument must be a variable name")
    value_var isa Symbol || error("interpolate() second argument must be a variable name")
    mode in (:constant, :linear, :spline) || error("interpolate() mode must be :constant, :linear, or :spline")

    return InterpolateInfo(time_var, value_var, mode)
end

"""
    parse_odin_block(block::Expr) -> Vector{OdinExpr}

Parse an entire odin block expression into a list of OdinExpr.
"""
function parse_odin_block(block::Expr)
    exprs = OdinExpr[]
    src = LineNumberNode(0, :unknown)

    function walk(e)
        if e isa LineNumberNode
            src = e
        elseif e isa Expr
            if e.head == :block
                for a in e.args
                    walk(a)
                end
            else
                result = parse_odin_expr(e, src)
                if result isa Vector
                    append!(exprs, result)
                else
                    push!(exprs, result)
                end
            end
        end
    end

    walk(block)
    return exprs
end
