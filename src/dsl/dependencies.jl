# Dependency graph construction and topological sorting for odin models.

"""
    find_dependencies(expr) -> Set{Symbol}

Extract all symbol references from an expression.
"""
function find_dependencies(expr)::Set{Symbol}
    deps = Set{Symbol}()
    _walk_deps!(deps, expr)
    return deps
end

function _walk_deps!(deps::Set{Symbol}, expr)
    if expr isa Symbol
        if !(expr in RESERVED_NAMES) && !(expr in (:nothing, :true, :false, :Inf, :NaN, :pi, :π))
            push!(deps, expr)
        end
    elseif expr isa Expr
        if expr.head == :call
            # Don't include function name as dependency
            fname = expr.args[1]
            if fname isa Symbol && (fname in MATH_FUNCTIONS || haskey(DISTRIBUTION_MAP, fname) || fname in SPECIAL_RHS)
                for a in expr.args[2:end]
                    _walk_deps!(deps, a)
                end
            elseif fname isa Symbol
                # Unknown function — include args
                for a in expr.args[2:end]
                    _walk_deps!(deps, a)
                end
            end
        elseif expr.head == :ref
            # Array access X[i] — X is a dependency, indices may be too
            _walk_deps!(deps, expr.args[1])
            for a in expr.args[2:end]
                _walk_deps!(deps, a)
            end
        elseif expr.head == :kw
            # keyword arg — only value is dependency
            _walk_deps!(deps, expr.args[2])
        else
            for a in expr.args
                _walk_deps!(deps, a)
            end
        end
    end
    # Numbers, strings, etc. have no dependencies
    return deps
end

"""
Dependency graph entry for a single equation.
"""
struct DepEntry
    name::Symbol
    expr_type::ExprType
    depends_on::Set{Symbol}
end

"""
    build_dependency_graph(exprs::Vector{OdinExpr}, classification::ModelClassification)

Build a dependency graph for all equations. Returns a vector of DepEntry
and a topologically sorted order.
"""
function build_dependency_graph(exprs::Vector{OdinExpr}, classification::ModelClassification)
    entries = DepEntry[]

    for ex in exprs
        # Skip dim declarations — they're metadata
        ex.type == EXPR_DIM && continue
        # Skip print statements — they don't define variables
        ex.type == EXPR_PRINT && continue

        deps = Set{Symbol}()
        if ex.type == EXPR_INITIAL
            rhs_data = ex.rhs
            rhs_expr = rhs_data isa NamedTuple ? rhs_data.rhs : rhs_data
            deps = find_dependencies(rhs_expr)
        elseif ex.type == EXPR_COMPARE
            cinfo = ex.rhs::CompareInfo
            for arg in cinfo.args
                union!(deps, find_dependencies(arg))
            end
        elseif ex.type == EXPR_INTERPOLATE
            iinfo = ex.rhs::InterpolateInfo
            push!(deps, iinfo.time_var)
            push!(deps, iinfo.value_var)
        elseif ex.type == EXPR_PARAMETER
            pinfo = ex.rhs::ParameterInfo
            if pinfo.default !== nothing
                union!(deps, find_dependencies(pinfo.default))
            end
        else
            union!(deps, find_dependencies(ex.rhs))
        end

        # Remove index variables and self-references for state vars
        setdiff!(deps, INDEX_VARIABLES)
        # Remove the variable's own name if it is a state variable (self-ref is ok for update/deriv/diffusion)
        if ex.type in (EXPR_UPDATE, EXPR_DERIV, EXPR_DIFFUSION)
            delete!(deps, ex.name)
        end

        push!(entries, DepEntry(ex.name, ex.type, deps))
    end

    return entries
end

"""
    topological_sort(entries::Vector{DepEntry}, state_vars::Vector{Symbol}) -> Vector{Symbol}

Topologically sort equations so dependencies come before dependents.
State variables and parameters are treated as available from the start.
"""
function topological_sort(entries::Vector{DepEntry}, available::Set{Symbol})
    # Build name → entry mapping (skip duplicates from deriv/update/initial pairs)
    # We only sort intermediate assignments
    to_sort = DepEntry[]
    for e in entries
        if e.expr_type == EXPR_ASSIGNMENT
            push!(to_sort, e)
        end
    end

    sorted = Symbol[]
    resolved = copy(available)
    remaining = Dict(e.name => e for e in to_sort)

    max_iter = length(remaining) + 1
    for _ in 1:max_iter
        isempty(remaining) && break
        progress = false
        for (name, entry) in remaining
            if issubset(entry.depends_on, resolved)
                push!(sorted, name)
                push!(resolved, name)
                delete!(remaining, name)
                progress = true
            end
        end
        progress || error("Circular dependency detected among: $(collect(keys(remaining)))")
    end

    return sorted
end
