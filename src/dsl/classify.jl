# Variable classification for odin models.
# Determines which variables are state, parameters, data, intermediates, etc.

"""Classification of a variable in the model."""
@enum VarRole begin
    VAR_STATE           # state variable (has deriv/update + initial)
    VAR_PARAMETER       # user-supplied parameter
    VAR_DATA            # observed data input
    VAR_INTERMEDIATE    # computed intermediate (equation)
    VAR_INTERPOLATED    # interpolated quantity
    VAR_OUTPUT          # output-only variable
end

"""Information about a single variable in the model."""
struct VarInfo
    name::Symbol
    role::VarRole
    is_array::Bool
    rank::Int               # 0 for scalar
    dims::Any               # dimension expression or nothing
    zero_every::Any         # zero_every period or nothing
end

"""Time type of the model."""
@enum TimeType begin
    TIME_CONTINUOUS     # uses deriv()
    TIME_DISCRETE       # uses update()
end

"""
Complete classification of all variables in an odin model.
"""
struct ModelClassification
    time_type::TimeType
    state_vars::Vector{Symbol}
    parameters::Dict{Symbol, ParameterInfo}
    data_vars::Vector{Symbol}
    intermediates::Vector{Symbol}
    interpolated::Dict{Symbol, InterpolateInfo}
    delayed::Dict{Symbol, DelayInfo}
    outputs::Vector{Symbol}
    comparisons::Dict{Symbol, CompareInfo}
    dims::Dict{Symbol, Any}            # variable => dimension expr
    zero_every::Dict{Symbol, Any}      # variable => period
    all_vars::Dict{Symbol, VarInfo}
    diffusion_vars::Set{Symbol}        # state vars with diffusion() terms
end

"""
    classify_variables(exprs::Vector{OdinExpr}) -> ModelClassification

Analyse parsed expressions and classify all variables.
"""
function classify_variables(exprs::Vector{OdinExpr})
    has_deriv = false
    has_update = false

    # Collect variable roles
    initial_vars = Dict{Symbol, Any}()
    deriv_vars = Set{Symbol}()
    update_vars = Set{Symbol}()
    diffusion_vars = Set{Symbol}()
    parameters = Dict{Symbol, ParameterInfo}()
    data_vars = Set{Symbol}()
    interpolated = Dict{Symbol, InterpolateInfo}()
    delayed = Dict{Symbol, DelayInfo}()
    outputs = Symbol[]
    comparisons = Dict{Symbol, CompareInfo}()
    dims = Dict{Symbol, Any}()
    zero_every = Dict{Symbol, Any}()
    assigned = Set{Symbol}()

    for ex in exprs
        if ex.type == EXPR_DERIV
            has_deriv = true
            push!(deriv_vars, ex.name)
        elseif ex.type == EXPR_UPDATE
            has_update = true
            push!(update_vars, ex.name)
        elseif ex.type == EXPR_DIFFUSION
            push!(diffusion_vars, ex.name)
        elseif ex.type == EXPR_INITIAL
            rhs_data = ex.rhs
            if rhs_data isa NamedTuple
                initial_vars[ex.name] = rhs_data.rhs
                if rhs_data.zero_every !== nothing
                    zero_every[ex.name] = rhs_data.zero_every
                end
            else
                initial_vars[ex.name] = rhs_data
            end
        elseif ex.type == EXPR_DIM
            dims[ex.name] = ex.rhs
        elseif ex.type == EXPR_OUTPUT
            ex.name in outputs || push!(outputs, ex.name)
        elseif ex.type == EXPR_PARAMETER
            parameters[ex.name] = ex.rhs
        elseif ex.type == EXPR_DATA
            push!(data_vars, ex.name)
        elseif ex.type == EXPR_INTERPOLATE
            interpolated[ex.name] = ex.rhs
        elseif ex.type == EXPR_DELAY
            delayed[ex.name] = ex.rhs
            push!(assigned, ex.name)
        elseif ex.type == EXPR_COMPARE
            comparisons[ex.name] = ex.rhs
        elseif ex.type == EXPR_ASSIGNMENT
            push!(assigned, ex.name)
        elseif ex.type == EXPR_PRINT
            # Print expressions don't define variables; nothing to classify
        end
    end

    # Determine time type
    has_deriv && has_update && error("Cannot mix deriv() and update() in the same model")
    !has_deriv && !has_update && error("Model must have at least one deriv() or update() equation")
    time_type = has_deriv ? TIME_CONTINUOUS : TIME_DISCRETE

    # State variables: those with both initial and deriv/update, in definition order
    dynamic_vars = time_type == TIME_CONTINUOUS ? deriv_vars : update_vars
    state_vars = Symbol[]
    seen = Set{Symbol}()
    for ex in exprs
        if ex.type == EXPR_INITIAL && ex.name in dynamic_vars && !(ex.name in seen)
            push!(state_vars, ex.name)
            push!(seen, ex.name)
        end
    end

    # Validate: every state var has both initial and dynamics
    for v in dynamic_vars
        v in keys(initial_vars) || error("Variable $v has dynamics but no initial()")
    end
    for v in keys(initial_vars)
        v in dynamic_vars || v in outputs || error("Variable $v has initial() but no deriv()/update()")
    end

    # Validate diffusion: every diffusion var must have a matching deriv
    for v in diffusion_vars
        v in deriv_vars || error("Variable $v has diffusion() but no deriv()")
    end
    !isempty(diffusion_vars) && has_update && error("Cannot use diffusion() with update() (discrete) models")
    !isempty(delayed) && has_update && error("Cannot use delay() with update() (discrete) models")

    # Intermediates: assigned but not state/parameter/data/interpolated
    intermediates = Symbol[]
    for v in assigned
        if !(v in keys(parameters)) && !(v in data_vars) && !(v in keys(interpolated)) && !(v in dynamic_vars)
            push!(intermediates, v)
        end
    end

    # Build VarInfo for all variables
    all_vars = Dict{Symbol, VarInfo}()
    for v in state_vars
        is_arr = haskey(dims, v)
        rank = is_arr ? _dim_rank(dims[v]) : 0
        ze = get(zero_every, v, nothing)
        all_vars[v] = VarInfo(v, VAR_STATE, is_arr, rank, get(dims, v, nothing), ze)
    end
    for (v, pinfo) in parameters
        is_arr = pinfo.rank > 0 || haskey(dims, v)
        rank = pinfo.rank > 0 ? pinfo.rank : (haskey(dims, v) ? _dim_rank(dims[v]) : 0)
        all_vars[v] = VarInfo(v, VAR_PARAMETER, is_arr, rank, get(dims, v, nothing), nothing)
    end
    for v in data_vars
        is_arr = haskey(dims, v)
        rank = is_arr ? _dim_rank(dims[v]) : 0
        all_vars[v] = VarInfo(v, VAR_DATA, is_arr, rank, get(dims, v, nothing), nothing)
    end
    for v in intermediates
        is_arr = haskey(dims, v)
        rank = is_arr ? _dim_rank(dims[v]) : 0
        all_vars[v] = VarInfo(v, VAR_INTERMEDIATE, is_arr, rank, get(dims, v, nothing), nothing)
    end
    for (v, _) in interpolated
        all_vars[v] = VarInfo(v, VAR_INTERPOLATED, false, 0, nothing, nothing)
    end
    for v in outputs
        is_arr = haskey(dims, v)
        rank = is_arr ? _dim_rank(dims[v]) : 0
        all_vars[v] = VarInfo(v, VAR_OUTPUT, is_arr, rank, get(dims, v, nothing), nothing)
    end

    return ModelClassification(
        time_type,
        state_vars,
        parameters,
        collect(data_vars),
        intermediates,
        interpolated,
        delayed,
        outputs,
        comparisons,
        dims,
        zero_every,
        all_vars,
        diffusion_vars,
    )
end

function _dim_rank(dim_expr)
    if dim_expr isa Number
        return 1
    elseif dim_expr isa Expr && dim_expr.head == :vect
        return length(dim_expr.args)
    elseif dim_expr isa Expr && dim_expr.head == :call && dim_expr.args[1] == :parameter
        # dim(X) = parameter(rank=N) — rank extracted from parameter info
        for arg in dim_expr.args[2:end]
            if arg isa Expr && arg.head == :kw && arg.args[1] == :rank
                return arg.args[2]
            end
        end
        return 1
    elseif dim_expr isa Symbol
        return 1
    else
        return 1
    end
end
