# Code generation: convert parsed odin model into Julia types and functions.
# This is the heart of the DSL compiler — it generates a DustSystemGenerator.

using Printf

const _PRINTF_CACHE = Dict{String, Printf.Format}()

"""Configurable output stream for print() statements. Defaults to stdout."""
const _PRINT_IO = Ref{IO}(stdout)

"""Runtime helper for print() statements in generated odin models."""
function _odin_printf(fmt::String, args...)
    f = get!(_PRINTF_CACHE, fmt) do
        Printf.Format(fmt)
    end
    Printf.format(_PRINT_IO[], f, args...)
    return nothing
end

# ── Array & layout helpers ────────────────────────────────────

"""Check whether any state variable is an array."""
function _has_arrays(cl::ModelClassification)
    any(v -> haskey(cl.dims, v), cl.state_vars)
end

"""Collect parameter symbols referenced in dimension expressions."""
function _collect_dim_params(dims::Dict{Symbol,Any}, cl::ModelClassification)
    params = Set{Symbol}()
    for (_, de) in dims
        _walk_dim_params!(params, de, cl)
    end
    return params
end

function _walk_dim_params!(params, expr, cl)
    if expr isa Symbol && haskey(cl.parameters, expr)
        push!(params, expr)
    elseif expr isa Expr && expr.head == :vect
        for a in expr.args; _walk_dim_params!(params, a, cl); end
    elseif expr isa Expr && expr.head == :call && expr.args[1] == :c
        for a in expr.args[2:end]; _walk_dim_params!(params, a, cl); end
    elseif expr isa Expr && expr.head == :block
        for a in expr.args
            a isa LineNumberNode && continue
            _walk_dim_params!(params, a, cl)
        end
    elseif expr isa Tuple
        for a in expr; _walk_dim_params!(params, a, cl); end
    end
end

"""Generate `dim_var = Int(pars.dim_var)` for all dimension parameters."""
function _gen_dim_stmts(dims, cl)
    stmts = Expr[]
    for p in sort(collect(_collect_dim_params(dims, cl)))
        push!(stmts, :($p = Int(pars.$p)))
    end
    return stmts
end

"""Total element count for a dim declaration (number, symbol, or [a,b,...])."""
function _dim_total(de)
    de isa Number && return de
    de isa Symbol && return de
    if de isa Expr && de.head == :vect
        parts = Any[_dim_total(a) for a in de.args]
        return length(parts) == 1 ? parts[1] : Expr(:call, :*, parts...)
    end
    # Handle c(a, b) calls from R-style dim declarations
    if de isa Expr && de.head == :call && de.args[1] == :c
        parts = Any[_dim_total(a) for a in de.args[2:end]]
        return length(parts) == 1 ? parts[1] : Expr(:call, :*, parts...)
    end
    # Handle tuples (n_age, n_vax)
    if de isa Tuple
        parts = Any[_dim_total(a) for a in de]
        return length(parts) == 1 ? parts[1] : Expr(:call, :*, parts...)
    end
    # Unwrap block expressions from quote ... end
    if de isa Expr && de.head == :block
        for a in de.args
            a isa LineNumberNode && continue
            return _dim_total(a)
        end
    end
    return de
end

"""Per-dimension sizes as a vector of expressions."""
function _dim_each(de)
    (de isa Number || de isa Symbol) && return Any[de]
    if de isa Expr && de.head == :vect
        return Any[_dim_total(a) for a in de.args]
    end
    # Handle c(a, b) calls
    if de isa Expr && de.head == :call && de.args[1] == :c
        return Any[_dim_total(a) for a in de.args[2:end]]
    end
    # Handle tuples
    if de isa Tuple
        return Any[_dim_total(a) for a in de]
    end
    # Unwrap block expressions
    if de isa Expr && de.head == :block
        for a in de.args
            a isa LineNumberNode && continue
            return _dim_each(a)
        end
    end
    return Any[de]
end

"""Generate state offset/size computation.

Returns `(stmts, offsets, sizes)` where `offsets[v]` is a Symbol like
`_off_S` and `sizes[v]` is a number or expression.
"""
function _gen_layout(state_vars, dims)
    stmts   = Expr[]
    offsets  = Dict{Symbol,Symbol}()
    sizes    = Dict{Symbol,Any}()
    for (i, v) in enumerate(state_vars)
        off = Symbol("_off_", v)
        sz  = haskey(dims, v) ? _dim_total(dims[v]) : 1
        if i == 1
            push!(stmts, :($off = 0))
        else
            pv = state_vars[i-1]
            push!(stmts, :($off = $(offsets[pv]) + $(sizes[pv])))
        end
        offsets[v] = off
        sizes[v]   = sz
    end
    return stmts, offsets, sizes
end

"""Generate state unpacking: views for array vars, locals for scalars."""
function _gen_unpack(state_vars, dims, offsets, sizes; src=:state)
    stmts = Expr[]
    for v in state_vars
        off = offsets[v]; sz = sizes[v]
        if haskey(dims, v)
            ds = _dim_each(dims[v])
            if length(ds) == 1
                push!(stmts, :($v = view($src, ($off + 1):($off + $sz))))
            else
                push!(stmts, :($v = reshape(view($src, ($off + 1):($off + $sz)), $(ds...))))
            end
        else
            push!(stmts, :($v = $src[$off + 1]))
        end
    end
    return stmts
end

"""Allocate local arrays for array intermediates."""
function _gen_intermediate_arrays(intermediates, dims; use_workspace::Bool=false)
    stmts = Expr[]
    for v in intermediates
        haskey(dims, v) || continue
        ds = _dim_each(dims[v])
        vq = QuoteNode(v)
        if use_workspace
            # Reuse pre-allocated arrays from model._workspace (zero-alloc hot path)
            # Check both existence AND size — dimensions may come from parameters
            # that change between calls (e.g. dim(E) = k_E).
            if length(ds) == 1
                dim_expr = ds[1]
                push!(stmts, :($v = let _dim = $dim_expr
                    if haskey(model._workspace, $vq)
                        _cached = model._workspace[$vq]::Vector{Float64}
                        if length(_cached) == _dim
                            _cached
                        else
                            _v = Vector{Float64}(undef, _dim)
                            model._workspace[$vq] = _v
                            _v
                        end
                    else
                        _v = Vector{Float64}(undef, _dim)
                        model._workspace[$vq] = _v
                        _v
                    end
                end))
            else
                ndim = length(ds)
                dims_tuple = Expr(:tuple, ds...)
                push!(stmts, :($v = let _dims = $dims_tuple
                    if haskey(model._workspace, $vq)
                        _cached = model._workspace[$vq]::Array{Float64, $ndim}
                        if size(_cached) == _dims
                            _cached
                        else
                            _v = Array{Float64}(undef, _dims...)
                            model._workspace[$vq] = _v
                            _v
                        end
                    else
                        _v = Array{Float64}(undef, _dims...)
                        model._workspace[$vq] = _v
                        _v
                    end
                end))
            end
        else
            if length(ds) == 1
                push!(stmts, :($v = Vector{Float64}(undef, $(ds[1]))))
            else
                push!(stmts, :($v = Array{Float64}(undef, $(ds...))))
            end
        end
    end
    return stmts
end

"""Find the loop bound for a reduction index by scanning array refs in the expression."""
function _find_reduction_bound(expr, idx::Symbol, classification)
    dims = classification.dims
    if expr isa Expr && expr.head == :ref
        arr_name = expr.args[1]
        if haskey(dims, arr_name)
            ds = _dim_each(dims[arr_name])
            # Find which position idx appears in the indexing
            for (k, a) in enumerate(expr.args[2:end])
                if a == idx && k <= length(ds)
                    return ds[k]
                end
            end
        end
    elseif expr isa Expr
        for a in expr.args
            r = _find_reduction_bound(a, idx, classification)
            r !== nothing && return r
        end
    end
    return nothing
end

"""Wrap statement in nested for loops for given indices and dim expression."""
function _wrap_loop(stmt, indices, dim_expr; range_bounds=Dict{Symbol,Tuple{Any,Any}}())
    isempty(indices) && return stmt
    ds = _dim_each(dim_expr)
    result = stmt
    for k in length(indices):-1:1
        idx = indices[k]
        if haskey(range_bounds, idx)
            lo, hi = range_bounds[idx]
            result = :(for $idx in $lo:$hi; $result; end)
        else
            bound = ds[min(k, length(ds))]
            result = :(for $idx in 1:$bound; $result; end)
        end
    end
    return result
end

"""Column-major linear index for multi-dim: `(j-1)*d1 + i`."""
function _linear_idx(indices, dim_expr)
    length(indices) == 1 && return indices[1]
    ds = _dim_each(dim_expr)
    expr = indices[1]
    stride = ds[1]
    for k in 2:length(indices)
        term = :($(indices[k]) - 1)
        expr = :($expr + $term * $stride)
        k < length(indices) && (stride = :($stride * $(ds[k])))
    end
    return expr
end

# ── Main entry point ──────────────────────────────────────────

"""
    generate_system(exprs, classification, phases)

Generate Julia code (as Expr) that defines a dust system type and its methods.
Returns a quote block that, when evaluated, creates a `DustSystemGenerator`.
"""
function generate_system(
    exprs::Vector{OdinExpr},
    classification::ModelClassification,
    phases::ModelPhases,
)
    model_name    = gensym("OdinModel")
    has_compare   = !isempty(classification.comparisons)
    has_output    = !isempty(classification.outputs)
    has_interp    = !isempty(classification.interpolated)
    is_continuous = classification.time_type == TIME_CONTINUOUS
    is_sde        = !isempty(classification.diffusion_vars)
    has_delay     = !isempty(classification.delayed)
    has_arr       = _has_arrays(classification)
    n_state_fixed = has_arr ? 0 : length(classification.state_vars)

    state_var_set = Set{Symbol}(classification.state_vars)

    param_names = Symbol[ex.name for ex in phases.create_eqs if ex.type == EXPR_PARAMETER]

    # Generate method bodies
    n_state_body     = _gen_n_state_body(classification)
    state_names_body = _gen_state_names_body(classification)
    initial_body     = _gen_initial(phases, classification, state_var_set)
    dynamics_body    = is_continuous ?
        _gen_rhs(phases, classification, state_var_set) :
        _gen_update(phases, classification, state_var_set)
    diffusion_body   = is_sde ?
        _gen_diffusion(phases, classification, state_var_set) : nothing
    compare_body     = has_compare ?
        _gen_compare(phases, classification, state_var_set) : nothing
    output_body      = has_output ?
        _gen_output(phases, classification, state_var_set) : nothing
    n_output_body    = has_output ?
        _gen_n_output_body(classification) : nothing
    output_names_body = has_output ?
        _gen_output_names_body(classification) : nothing
    setup_pars_body  = has_interp ?
        _gen_setup_pars(classification) : nothing

    has_zero_every = !isempty(classification.zero_every)
    zero_every_body = has_zero_every ?
        _gen_zero_every_body(classification) : nothing

    # Generate symbolic Jacobian if possible (continuous models only)
    symbolic_jac_code = nothing
    if is_continuous
        symbolic_jac_code = _gen_symbolic_jacobian(phases, classification,
                                                    state_var_set, model_name)
    end

    code = quote
        mutable struct $model_name <: Odin.AbstractOdinModel
            n_state::Int
            state_names::Vector{Symbol}
            parameter_names::Vector{Symbol}
            is_continuous::Bool
            is_sde::Bool
            has_compare::Bool
            has_output::Bool
            has_interpolation::Bool
            has_delay::Bool
            _workspace::Dict{Symbol, Array}
        end

        function Odin._odin_n_state(model::$model_name, pars)
            $n_state_body
        end

        function Odin._odin_state_names(model::$model_name, pars)
            $state_names_body
        end

        $(if has_zero_every
            quote
                function Odin._odin_zero_every(model::$model_name, pars)
                    $zero_every_body
                end
            end
        else
            nothing
        end)

        function Odin._odin_initial!(model::$model_name, state::AbstractVector,
                                     pars, rng::Odin.Random.AbstractRNG)
            @inbounds $initial_body
            return nothing
        end

        $(if is_continuous
            quote
                function Odin._odin_rhs!(model::$model_name, dstate::AbstractVector{T},
                                         state::AbstractVector{T}, pars, time) where {T}
                    @inbounds $dynamics_body
                    return nothing
                end
            end
        else
            quote
                function Odin._odin_update!(model::$model_name, state_next::AbstractVector{T},
                                            state::AbstractVector{T}, pars,
                                            time::T, dt::T, rng::Odin.Random.AbstractRNG) where {T}
                    @inbounds $dynamics_body
                    return nothing
                end
            end
        end)

        $(if is_sde
            quote
                function Odin._odin_diffusion!(model::$model_name, noise_out::AbstractVector{T},
                                               state::AbstractVector{T}, pars, time) where {T}
                    @inbounds $diffusion_body
                    return nothing
                end
            end
        else
            nothing
        end)

        $(if has_compare
            quote
                function Odin._odin_compare_data(model::$model_name, state::AbstractVector,
                                                 pars, data::NamedTuple, time)
                    @inbounds $compare_body
                end
            end
        else
            nothing
        end)

        $(if has_output
            quote
                function Odin._odin_n_output(model::$model_name, pars)
                    $n_output_body
                end
                function Odin._odin_output_names(model::$model_name, pars)
                    $output_names_body
                end
                function Odin._odin_output!(model::$model_name, output_buf::AbstractVector{T},
                                            state::AbstractVector{T}, pars, time) where {T}
                    @inbounds $output_body
                    return nothing
                end
            end
        else
            nothing
        end)

        $(if has_interp
            quote
                function Odin._odin_setup_pars(model::$model_name, pars)
                    $setup_pars_body
                end
            end
        else
            nothing
        end)

        $(if has_delay
            delay_tau_body = _gen_delay_tau_body(classification)
            quote
                function Odin._odin_delay_tau_values(model::$model_name, pars)
                    $delay_tau_body
                end
            end
        else
            nothing
        end)

        $(if symbolic_jac_code !== nothing
            symbolic_jac_code
        else
            nothing
        end)

        Odin.DustSystemGenerator(
            $model_name(
                $n_state_fixed,
                $(QuoteNode(classification.state_vars)),
                $(QuoteNode(param_names)),
                $is_continuous,
                $is_sde,
                $has_compare,
                $has_output,
                $has_interp,
                $has_delay,
                Dict{Symbol, Array}(),
            )
        )
    end

    return code
end

# ── n_state / state_names generation ──────────────────────────

function _gen_n_state_body(cl::ModelClassification)
    _has_arrays(cl) || return :(return $(length(cl.state_vars)))
    stmts = Expr[]
    append!(stmts, _gen_dim_stmts(cl.dims, cl))
    ls, offs, szs = _gen_layout(cl.state_vars, cl.dims)
    append!(stmts, ls)
    last = cl.state_vars[end]
    push!(stmts, :(return $(offs[last]) + $(szs[last])))
    return Expr(:block, stmts...)
end

function _gen_state_names_body(cl::ModelClassification)
    _has_arrays(cl) || return :(return $(QuoteNode(cl.state_vars)))
    stmts = Expr[]
    append!(stmts, _gen_dim_stmts(cl.dims, cl))
    push!(stmts, :(_names = Symbol[]))
    for v in cl.state_vars
        vstr = string(v)
        if haskey(cl.dims, v)
            ds = _dim_each(cl.dims[v])
            if length(ds) == 1
                push!(stmts, :(for _i in 1:$(ds[1])
                    push!(_names, Symbol($vstr, "[", _i, "]"))
                end))
            else
                sz = _dim_total(cl.dims[v])
                push!(stmts, :(for _i in 1:$sz
                    push!(_names, Symbol($vstr, "[", _i, "]"))
                end))
            end
        else
            push!(stmts, :(push!(_names, $(QuoteNode(v)))))
        end
    end
    push!(stmts, :(return _names))
    return Expr(:block, stmts...)
end

"""Generate zero_every info: returns Dict{UnitRange{Int}, Int} mapping state indices to period."""
function _gen_zero_every_body(cl::ModelClassification)
    stmts = Expr[]
    dims = cl.dims
    append!(stmts, _gen_dim_stmts(dims, cl))
    ls, offs, szs = _gen_layout(cl.state_vars, dims)
    append!(stmts, ls)

    push!(stmts, :(_ze = Dict{UnitRange{Int}, Int}()))
    for (v, period) in cl.zero_every
        off = offs[v]
        sz = szs[v]
        # period might be an expression referencing parameters
        period_expr = period isa Symbol && haskey(cl.parameters, period) ?
            :(Int(pars.$period)) : :(Int($period))
        if haskey(dims, v)
            push!(stmts, :(_ze[($off + 1):($off + $sz)] = $period_expr))
        else
            push!(stmts, :(_ze[($off + 1):($off + 1)] = $period_expr))
        end
    end
    push!(stmts, :(return _ze))
    return Expr(:block, stmts...)
end

# ── Output support generation ─────────────────────────────────

"""Check whether any output variable is an array."""
function _has_output_arrays(cl::ModelClassification)
    any(v -> haskey(cl.dims, v), cl.outputs)
end

"""Generate body for _odin_n_output: returns number of output slots."""
function _gen_n_output_body(cl::ModelClassification)
    dims = cl.dims
    _has_output_arrays(cl) || return :(return $(length(cl.outputs)))
    stmts = Expr[]
    append!(stmts, _gen_dim_stmts(dims, cl))
    # Compute total output size
    total = Expr(:call, :+)
    for v in cl.outputs
        if haskey(dims, v)
            push!(total.args, _dim_total(dims[v]))
        else
            push!(total.args, 1)
        end
    end
    push!(stmts, :(return $total))
    return Expr(:block, stmts...)
end

"""Generate body for _odin_output_names."""
function _gen_output_names_body(cl::ModelClassification)
    dims = cl.dims
    _has_output_arrays(cl) || return :(return $(QuoteNode(cl.outputs)))
    stmts = Expr[]
    append!(stmts, _gen_dim_stmts(dims, cl))
    push!(stmts, :(_names = Symbol[]))
    for v in cl.outputs
        vstr = string(v)
        if haskey(dims, v)
            ds = _dim_each(dims[v])
            if length(ds) == 1
                push!(stmts, :(for _i in 1:$(ds[1])
                    push!(_names, Symbol($vstr, "[", _i, "]"))
                end))
            else
                sz = _dim_total(dims[v])
                push!(stmts, :(for _i in 1:$sz
                    push!(_names, Symbol($vstr, "[", _i, "]"))
                end))
            end
        else
            push!(stmts, :(push!(_names, $(QuoteNode(v)))))
        end
    end
    push!(stmts, :(return _names))
    return Expr(:block, stmts...)
end

"""
Generate body for _odin_output!: computes output values and writes into output_buf.
output_buf has length n_output (one slot per output variable element).
Handles both `output(x) = true` (flag: copy from state/intermediate) and
`output(x) = expr` (expression: compute and store).
"""
function _is_output_flag(rhs)
    rhs === true && return true
    rhs == :true && return true
    if rhs isa Expr && rhs.head == :block
        for a in rhs.args
            a isa LineNumberNode && continue
            return _is_output_flag(a)
        end
    end
    return false
end

function _gen_output(phases, cl, sv_set)
    stmts = Expr[]
    dims = cl.dims

    append!(stmts, _gen_dim_stmts(dims, cl))

    # Layout for state variables (to unpack state)
    ls, offs, szs = _gen_layout(cl.state_vars, dims)
    append!(stmts, ls)
    append!(stmts, _gen_unpack(cl.state_vars, dims, offs, szs))

    # Collect symbols needed by output expressions
    output_needs = Set{Symbol}()
    for ex in phases.output_eqs
        if _is_output_flag(ex.rhs)
            # Flag form: output(x) = true — need the variable x itself
            push!(output_needs, ex.name)
        else
            union!(output_needs, find_dependencies(ex.rhs))
        end
    end

    # Transitive closure on intermediates
    intermediate_set = Set{Symbol}(cl.intermediates)
    changed = true
    while changed
        changed = false
        for ex in phases.dynamic_eqs
            ex.type == EXPR_ASSIGNMENT || continue
            if ex.name in output_needs && ex.name in intermediate_set
                deps = find_dependencies(ex.rhs)
                for d in deps
                    if d in intermediate_set && !(d in output_needs)
                        push!(output_needs, d)
                        changed = true
                    end
                end
            end
        end
    end

    # Allocate needed intermediate arrays
    needed_intermediates = [v for v in cl.intermediates if v in output_needs]
    for v in needed_intermediates
        haskey(dims, v) || continue
        ds = _dim_each(dims[v])
        if length(ds) == 1
            push!(stmts, :($v = Vector{Float64}(undef, $(ds[1]))))
        else
            push!(stmts, :($v = Array{Float64}(undef, $(ds...))))
        end
    end

    # Evaluate needed intermediates (from dynamic_eqs, in order)
    for ex in phases.dynamic_eqs
        ex.type == EXPR_ASSIGNMENT || continue
        ex.name in output_needs || continue
        rj = _translate_expr(ex.rhs, cl, sv_set, :pars)
        if !isempty(ex.indices) && haskey(dims, ex.name)
            assign = Expr(:(=), Expr(:ref, ex.name, ex.indices...), rj)
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :($(ex.name) = $rj))
        end
    end

    # Layout for output variables
    out_ls, out_offs, out_szs = _gen_layout(cl.outputs, dims)
    append!(stmts, out_ls)

    # Write output values into output_buf
    for ex in phases.output_eqs
        out_off = out_offs[ex.name]
        if _is_output_flag(ex.rhs)
            # Flag form: copy value of variable (state or intermediate)
            src_sym = ex.name
            if haskey(dims, ex.name)
                # Array flag: copy all elements
                out_sz = out_szs[ex.name]
                if src_sym in Set(cl.state_vars) && haskey(offs, src_sym)
                    # Copy from state vector offsets
                    st_off = offs[src_sym]
                    push!(stmts, :(for _k in 1:$out_sz
                        output_buf[$out_off + _k] = state[$st_off + _k]
                    end))
                else
                    # Copy from intermediate array
                    push!(stmts, :(for _k in 1:$out_sz
                        output_buf[$out_off + _k] = $src_sym[_k]
                    end))
                end
            else
                # Scalar flag
                push!(stmts, :(output_buf[$out_off + 1] = $src_sym))
            end
        else
            # Expression form
            rj = _translate_expr(ex.rhs, cl, sv_set, :pars)
            if !isempty(ex.indices) && haskey(dims, ex.name)
                li = _linear_idx(ex.indices, dims[ex.name])
                assign = :(output_buf[$out_off + $li] = convert(T, $rj))
                push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
            else
                push!(stmts, :(output_buf[$out_off + 1] = convert(T, $rj)))
            end
        end
    end

    return Expr(:block, stmts...)
end

# ── Interpolation setup generation ────────────────────────────

"""
Generate body for _odin_setup_pars: builds interpolators from parameter arrays
and returns augmented pars NamedTuple with _interp_X closure fields.
"""
function _gen_setup_pars(cl::ModelClassification)
    stmts = Expr[]
    interp_names = Symbol[]
    for (varname, iinfo) in cl.interpolated
        interp_sym = Symbol(:_interp_, varname)
        push!(interp_names, interp_sym)
        time_ref = :(pars.$(iinfo.time_var))
        value_ref = :(pars.$(iinfo.value_var))
        mode = QuoteNode(iinfo.mode)
        push!(stmts, :($interp_sym = Odin.build_interpolator(
            Base.collect(Float64, $time_ref),
            Base.collect(Float64, $value_ref),
            $mode)))
    end
    # Build merge expression: merge(pars, (_interp_beta = _interp_beta, ...))
    nt_pairs = [Expr(:(=), s, s) for s in interp_names]
    nt_expr = Expr(:tuple, nt_pairs...)
    push!(stmts, :(return merge(pars, $nt_expr)))
    return Expr(:block, stmts...)
end

# ── Body generators ───────────────────────────────────────────

function _gen_delay_tau_body(cl::ModelClassification)
    stmts = Expr[]
    push!(stmts, :(_taus = Float64[]))
    for (name, dinfo) in cl.delayed
        tau_expr = _translate_expr(dinfo.tau, cl, Set{Symbol}(cl.state_vars), :pars)
        push!(stmts, :(push!(_taus, Float64($tau_expr))))
    end
    push!(stmts, :(return _taus))
    return Expr(:block, stmts...)
end

function _gen_initial(phases, cl, sv_set)
    stmts = Expr[]
    dims = cl.dims
    append!(stmts, _gen_dim_stmts(dims, cl))
    ls, offs, _ = _gen_layout(cl.state_vars, dims)
    append!(stmts, ls)

    # Use eltype(state) for conversion (supports Dual numbers from ForwardDiff)
    pushfirst!(stmts, :(T = eltype(state)))

    for ex in phases.initial_eqs
        ex.type == EXPR_INITIAL || continue
        rd = ex.rhs
        rhs_e = rd isa NamedTuple ? rd.rhs : rd
        off = offs[ex.name]
        rj = _translate_expr(rhs_e, cl, sv_set, :pars)

        if !isempty(ex.indices) && haskey(dims, ex.name)
            li = _linear_idx(ex.indices, dims[ex.name])
            assign = :(state[$off + $li] = convert(T, $rj))
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :(state[$off + 1] = convert(T, $rj)))
        end
    end
    return Expr(:block, stmts...)
end

"""Generate print statements for all EXPR_PRINT expressions in the dynamic phase."""
function _gen_print_stmts(phases, cl, sv_set)
    stmts = Expr[]
    for ex in phases.dynamic_eqs
        ex.type == EXPR_PRINT || continue
        pinfo = ex.rhs::PrintInfo

        printf_fmt = pinfo.format_string
        for (var, fmt) in zip(pinfo.variables, pinfo.formats)
            printf_fmt = replace(printf_fmt, Regex("\\{$(var)(?:\\s*;\\s*[^}]*)?\\}") => fmt; count=1)
        end
        printf_fmt *= "\n"

        arg_exprs = Any[_translate_expr(var, cl, sv_set, :pars) for var in pinfo.variables]
        print_call = :(Odin._odin_printf($printf_fmt, $(arg_exprs...)))

        if pinfo.condition !== nothing
            cond = _translate_expr(pinfo.condition, cl, sv_set, :pars)
            push!(stmts, Expr(:if, cond, print_call))
        else
            push!(stmts, print_call)
        end
    end
    return stmts
end

function _gen_rhs(phases, cl, sv_set)
    stmts = Expr[]
    dims = cl.dims
    append!(stmts, _gen_dim_stmts(dims, cl))
    ls, offs, szs = _gen_layout(cl.state_vars, dims)
    append!(stmts, ls)
    append!(stmts, _gen_unpack(cl.state_vars, dims, offs, szs))
    append!(stmts, _gen_intermediate_arrays(cl.intermediates, dims; use_workspace=true))

    # Intermediates
    for ex in phases.dynamic_eqs
        ex.type == EXPR_ASSIGNMENT || continue
        rj = _translate_expr(ex.rhs, cl, sv_set, :pars)
        if !isempty(ex.indices) && haskey(dims, ex.name)
            assign = Expr(:(=), Expr(:ref, ex.name, ex.indices...), rj)
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :($(ex.name) = $rj))
        end
    end

    # Delay evaluations
    for ex in phases.dynamic_eqs
        ex.type == EXPR_DELAY || continue
        dinfo = ex.rhs::DelayInfo
        state_sym = dinfo.expr
        tau_expr = _translate_expr(dinfo.tau, cl, sv_set, :pars)
        state_off = offs[state_sym]
        state_idx = state_off isa Integer ? state_off + 1 : :($state_off + 1)
        push!(stmts, :(begin
            _dde_hist = (model._workspace[:_dde_history]::Vector{Odin.DDEHistory{T}})[1]
            $(ex.name) = Odin.dde_history_eval(_dde_hist, time - $tau_expr, $state_idx)
        end))
    end

    # Derivatives
    for ex in phases.dynamic_eqs
        ex.type == EXPR_DERIV || continue
        off = offs[ex.name]
        rj  = _translate_expr(ex.rhs, cl, sv_set, :pars)
        if !isempty(ex.indices) && haskey(dims, ex.name)
            li = _linear_idx(ex.indices, dims[ex.name])
            assign = :(dstate[$off + $li] = $rj)
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :(dstate[$off + 1] = $rj))
        end
    end

    append!(stmts, _gen_print_stmts(phases, cl, sv_set))

    return Expr(:block, stmts...)
end

function _gen_diffusion(phases, cl, sv_set)
    stmts = Expr[]
    dims = cl.dims
    append!(stmts, _gen_dim_stmts(dims, cl))
    ls, offs, szs = _gen_layout(cl.state_vars, dims)
    append!(stmts, ls)
    append!(stmts, _gen_unpack(cl.state_vars, dims, offs, szs))

    # Zero out noise_out first (states without diffusion get zero noise)
    push!(stmts, :(fill!(noise_out, zero(T))))

    # Intermediates needed by diffusion equations
    for ex in phases.diffusion_eqs
        ex.type == EXPR_ASSIGNMENT || continue
        rj = _translate_expr(ex.rhs, cl, sv_set, :pars)
        if !isempty(ex.indices) && haskey(dims, ex.name)
            assign = Expr(:(=), Expr(:ref, ex.name, ex.indices...), rj)
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :($(ex.name) = $rj))
        end
    end

    # Diffusion coefficients
    for ex in phases.diffusion_eqs
        ex.type == EXPR_DIFFUSION || continue
        off = offs[ex.name]
        rj  = _translate_expr(ex.rhs, cl, sv_set, :pars)
        if !isempty(ex.indices) && haskey(dims, ex.name)
            li = _linear_idx(ex.indices, dims[ex.name])
            assign = :(noise_out[$off + $li] = $rj)
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :(noise_out[$off + 1] = $rj))
        end
    end
    return Expr(:block, stmts...)
end

function _gen_update(phases, cl, sv_set)
    stmts = Expr[]
    dims = cl.dims
    append!(stmts, _gen_dim_stmts(dims, cl))
    ls, offs, szs = _gen_layout(cl.state_vars, dims)
    append!(stmts, ls)
    append!(stmts, _gen_unpack(cl.state_vars, dims, offs, szs))
    append!(stmts, _gen_intermediate_arrays(cl.intermediates, dims; use_workspace=true))

    # Intermediates
    for ex in phases.dynamic_eqs
        ex.type == EXPR_ASSIGNMENT || continue
        rj = _translate_expr(ex.rhs, cl, sv_set, :pars)
        if !isempty(ex.indices) && haskey(dims, ex.name)
            assign = Expr(:(=), Expr(:ref, ex.name, ex.indices...), rj)
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :($(ex.name) = $rj))
        end
    end

    # Updates
    for ex in phases.dynamic_eqs
        ex.type == EXPR_UPDATE || continue
        off = offs[ex.name]
        rj  = _translate_expr(ex.rhs, cl, sv_set, :pars)
        if !isempty(ex.indices) && haskey(dims, ex.name)
            li = _linear_idx(ex.indices, dims[ex.name])
            assign = :(state_next[$off + $li] = convert(T, $rj))
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :(state_next[$off + 1] = convert(T, $rj)))
        end
    end

    append!(stmts, _gen_print_stmts(phases, cl, sv_set))

    return Expr(:block, stmts...)
end

function _gen_compare(phases, cl, sv_set)
    stmts = Expr[]
    dims = cl.dims
    # Make dt available as a local (it's in pars via _merge_pars)
    push!(stmts, :(dt = pars.dt))
    append!(stmts, _gen_dim_stmts(dims, cl))
    ls, offs, szs = _gen_layout(cl.state_vars, dims)
    append!(stmts, ls)
    append!(stmts, _gen_unpack(cl.state_vars, dims, offs, szs))

    # Only include intermediates that the compare expressions depend on.
    # Collect symbols needed by compare expressions via transitive closure.
    compare_needs = Set{Symbol}()
    for ex in phases.compare_eqs
        cinfo = ex.rhs::CompareInfo
        for arg in cinfo.args
            union!(compare_needs, find_dependencies(arg))
        end
    end
    # Transitive closure: intermediates that other needed intermediates depend on
    changed = true
    intermediate_set = Set{Symbol}(cl.intermediates)
    while changed
        changed = false
        for ex in phases.dynamic_eqs
            ex.type == EXPR_ASSIGNMENT || continue
            if ex.name in compare_needs && ex.name in intermediate_set
                deps = find_dependencies(ex.rhs)
                for d in deps
                    if d in intermediate_set && !(d in compare_needs)
                        push!(compare_needs, d)
                        changed = true
                    end
                end
            end
        end
    end

    # Allocate only needed intermediate arrays
    needed_intermediates = [v for v in cl.intermediates if v in compare_needs]
    for v in needed_intermediates
        haskey(dims, v) || continue
        ds = _dim_each(dims[v])
        if length(ds) == 1
            push!(stmts, :($v = Vector{Float64}(undef, $(ds[1]))))
        else
            push!(stmts, :($v = Array{Float64}(undef, $(ds...))))
        end
    end

    # Only evaluate needed intermediates
    for ex in phases.dynamic_eqs
        ex.type == EXPR_ASSIGNMENT || continue
        ex.name in compare_needs || continue
        rj = _translate_expr(ex.rhs, cl, sv_set, :pars)
        if !isempty(ex.indices) && haskey(dims, ex.name)
            assign = Expr(:(=), Expr(:ref, ex.name, ex.indices...), rj)
            push!(stmts, _wrap_loop(assign, ex.indices, dims[ex.name]; range_bounds=ex.range_bounds))
        else
            push!(stmts, :($(ex.name) = $rj))
        end
    end

    push!(stmts, :(ll = 0.0))

    # Map distribution names to fast inline logpdf functions
    _FAST_LOGPDF = Dict{Symbol, Symbol}(
        :Poisson => :_logpdf_poisson,
        :Normal => :_logpdf_normal,
        :Binomial => :_logpdf_binomial,
        :NegativeBinomial => :_logpdf_negbinomial,
        :NegBinomial => :_logpdf_negbinomial,
        :Gamma => :_logpdf_gamma,
        :Exponential => :_logpdf_exponential,
        :Beta => :_logpdf_beta,
        :Uniform => :_logpdf_uniform,
    )

    for ex in phases.compare_eqs
        cinfo = ex.rhs::CompareInfo
        args_julia = Any[_translate_expr(a, cl, sv_set, :pars) for a in cinfo.args]

        is_array_compare = !isempty(ex.indices) && haskey(dims, ex.name)

        if is_array_compare
            # Array comparison: cases[i] ~ Poisson(lambda[i])
            # Index into data field: data.cases[i]
            data_val = Expr(:ref, :(data.$(ex.name)), ex.indices...)
        else
            # Scalar comparison: obs ~ Poisson(lambda)
            data_val = :(data.$(ex.name))
        end

        if haskey(_FAST_LOGPDF, cinfo.distribution)
            # Use allocation-free inline logpdf
            fast_fn = _FAST_LOGPDF[cinfo.distribution]
            logpdf_stmt = :(ll += Odin.$fast_fn($(args_julia...), $data_val))
        else
            # Fallback to Distributions.jl for unsupported distributions
            dist_entry = DISTRIBUTION_MAP[cinfo.distribution]
            dist_expr = :($(dist_entry.dist)($(args_julia...)))
            logpdf_stmt = :(ll += Odin.Distributions.logpdf($dist_expr, $data_val))
        end

        if is_array_compare
            push!(stmts, _wrap_loop(logpdf_stmt, ex.indices, dims[ex.name];
                                     range_bounds=ex.range_bounds))
        else
            push!(stmts, logpdf_stmt)
        end
    end
    push!(stmts, :(return ll))
    return Expr(:block, stmts...)
end

# ── Expression translator ─────────────────────────────────────

"""
    _translate_expr(expr, classification, state_var_set, pars_sym)

Translate an odin expression into Julia code, replacing parameter references
with `pars.name` and state variable references with local variable names.
`state_var_set` is a `Set{Symbol}` of state variable names (already unpacked).
"""
function _translate_expr(expr, classification, state_var_set::Set{Symbol}, pars_sym::Symbol;
                         dim_params::Set{Symbol}=_collect_dim_params(classification.dims, classification))
    if expr isa LineNumberNode
        return expr
    elseif expr isa Number || expr isa Bool
        return expr
    elseif expr isa Symbol
        if expr == :time
            return :time
        elseif expr == :dt
            return :dt
        elseif expr == :pi || expr == :π
            return π
        elseif expr in INDEX_VARIABLES
            return expr  # loop index variable
        elseif expr in dim_params
            return expr  # dim parameter — already local Int via _gen_dim_stmts
        elseif haskey(classification.parameters, expr)
            return :($pars_sym.$expr)
        elseif expr in state_var_set
            return expr  # already unpacked as local or view
        elseif expr in classification.data_vars
            return :(data.$expr)
        elseif haskey(classification.interpolated, expr)
            # Interpolated variable: call the stored interpolator closure
            interp_sym = Symbol(:_interp_, expr)
            return :($pars_sym.$interp_sym(time))
        else
            return expr  # intermediate variable or dim variable
        end
    elseif expr isa Expr
        if expr.head == :call
            fname = expr.args[1]
            if haskey(DISTRIBUTION_MAP, fname)
                dist_entry = DISTRIBUTION_MAP[fname]
                args = Any[_translate_expr(a, classification, state_var_set, pars_sym) for a in expr.args[2:end]]

                # Use fast inline samplers for common distributions
                if fname == :Binomial
                    n_arg = :(round(Int, $(args[1])))
                    p_arg = :(Float64($(args[2])))
                    return :(Odin._rand_binomial(rng, $n_arg, $p_arg))
                elseif fname == :Poisson
                    return :(Odin._rand_poisson(rng, $(args[1])))
                elseif fname == :Normal
                    return :(Odin._rand_normal(rng, $(args[1]), $(args[2])))
                elseif fname == :Exponential
                    return :(Odin._rand_exponential(rng, $(args[1])))
                elseif fname == :Gamma
                    return :(Odin._rand_gamma(rng, $(args[1]), $(args[2])))
                elseif fname == :NegativeBinomial || fname == :NegBinomial
                    n_arg = :(round(Int, $(args[1])))
                    p_arg = :(Float64($(args[2])))
                    return :(Odin._rand_nbinom(rng, $n_arg, $p_arg))
                elseif fname == :Multinomial
                    # Multinomial(n, prob_vector) → _rand_multinomial(rng, n, prob)
                    n_arg = :(round(Int, $(args[1])))
                    return :(Odin._rand_multinomial(rng, $n_arg, $(args[2])))
                else
                    # Fall back to Distributions.jl for less common distributions
                    dist_constructor = dist_entry.dist
                    if fname in (:Hypergeometric, :BetaBinomial)
                        args[1] = :(round(Int, $(args[1])))
                    end
                    return :(Odin.Random.rand(rng, $dist_constructor($(args...))))
                end
            elseif fname == :ifelse || fname == :if_else
                cond = _translate_expr(expr.args[2], classification, state_var_set, pars_sym)
                t = _translate_expr(expr.args[3], classification, state_var_set, pars_sym)
                f = _translate_expr(expr.args[4], classification, state_var_set, pars_sym)
                return :(ifelse($cond, $t, $f))
            elseif fname in REDUCTION_FUNCTIONS && length(expr.args) == 3 && expr.args[2] isa Symbol && expr.args[2] in INDEX_VARIABLES
                # Reduction: sum(j, expr) → loop accumulation
                idx = expr.args[2]
                body_expr = _translate_expr(expr.args[3], classification, state_var_set, pars_sym)
                # Find loop bound from array references in expression
                bound = _find_reduction_bound(expr.args[3], idx, classification)
                init_val = fname == :sum ? :(zero(Float64)) :
                           fname == :prod ? :(one(Float64)) :
                           fname == :min ? :(Inf) : :(-Inf)
                op = fname == :sum ? :+ : fname == :prod ? :* : fname == :min ? :min : :max
                acc = gensym(:_acc)
                return quote
                    let $acc = $init_val
                        for $idx in 1:$bound
                            $acc = $op($acc, $body_expr)
                        end
                        $acc
                    end
                end
            else
                new_args = Any[fname]
                for a in expr.args[2:end]
                    push!(new_args, _translate_expr(a, classification, state_var_set, pars_sym))
                end
                return Expr(:call, new_args...)
            end
        elseif expr.head == :if
            cond = _translate_expr(expr.args[1], classification, state_var_set, pars_sym)
            t = _translate_expr(expr.args[2], classification, state_var_set, pars_sym)
            if length(expr.args) >= 3
                f = _translate_expr(expr.args[3], classification, state_var_set, pars_sym)
                return :(ifelse($cond, $t, $f))
            else
                return :(ifelse($cond, $t, zero($t)))
            end
        elseif expr.head == :ref
            arr = _translate_expr(expr.args[1], classification, state_var_set, pars_sym)
            indices = Any[_translate_expr(a, classification, state_var_set, pars_sym) for a in expr.args[2:end]]
            return Expr(:ref, arr, indices...)
        elseif expr.head == :comparison
            new_args = Any[]
            for (i, a) in enumerate(expr.args)
                if i % 2 == 1
                    push!(new_args, _translate_expr(a, classification, state_var_set, pars_sym))
                else
                    push!(new_args, a)
                end
            end
            return Expr(:comparison, new_args...)
        elseif expr.head == :&&
            l = _translate_expr(expr.args[1], classification, state_var_set, pars_sym)
            r = _translate_expr(expr.args[2], classification, state_var_set, pars_sym)
            return Expr(:&&, l, r)
        elseif expr.head == :||
            l = _translate_expr(expr.args[1], classification, state_var_set, pars_sym)
            r = _translate_expr(expr.args[2], classification, state_var_set, pars_sym)
            return Expr(:||, l, r)
        elseif expr.head == :block
            new_args = Any[]
            for a in expr.args
                if !(a isa LineNumberNode)
                    push!(new_args, _translate_expr(a, classification, state_var_set, pars_sym))
                end
            end
            return length(new_args) == 1 ? new_args[1] : Expr(:block, new_args...)
        else
            new_args = Any[]
            for a in expr.args
                a isa LineNumberNode && continue
                push!(new_args, _translate_expr(a, classification, state_var_set, pars_sym))
            end
            return Expr(expr.head, new_args...)
        end
    elseif expr isa QuoteNode
        return expr.value
    else
        return expr
    end
end
