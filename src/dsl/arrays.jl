# Array handling for odin DSL.
# Manages dimension declarations and array index iteration.

"""
    resolve_array_dims(dims::Dict{Symbol,Any}, parameters::Dict{Symbol,ParameterInfo})

Resolve dim() declarations into concrete dimension info.
For statically known dims, returns integers; for parameter-dependent, returns symbols.
"""
function resolve_array_dims(dims::Dict{Symbol, Any}, parameters::Dict{Symbol, ParameterInfo})
    resolved = Dict{Symbol, Any}()
    for (name, dim_expr) in dims
        resolved[name] = _resolve_dim(dim_expr, parameters)
    end
    return resolved
end

function _resolve_dim(dim_expr, parameters)
    if dim_expr isa Integer
        return (dim_expr,)
    elseif dim_expr isa Symbol
        return (dim_expr,)
    elseif dim_expr isa Expr
        if dim_expr.head == :vect || dim_expr.head == :tuple
            return tuple(dim_expr.args...)
        elseif dim_expr.head == :call
            if dim_expr.args[1] == :parameter
                return :dynamic  # resolved at runtime
            elseif dim_expr.args[1] == :c
                return tuple(dim_expr.args[2:end]...)
            end
        end
    end
    return (dim_expr,)
end

"""
    shared_dim_groups(dims::Dict{Symbol,Any}) -> Dict{Any, Vector{Symbol}}

Group variables that share the same dimension expression (from `dim(X, Y) = N`).
"""
function shared_dim_groups(dims::Dict{Symbol, Any})
    groups = Dict{Any, Vector{Symbol}}()
    for (name, dim_expr) in dims
        key = dim_expr
        v = get!(Vector{Symbol}, groups, key)
        push!(v, name)
    end
    return groups
end
