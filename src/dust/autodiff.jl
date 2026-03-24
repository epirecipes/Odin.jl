# Reverse-mode AD fallback for VJP computation when symbolic
# differentiation is not available.
#
# Uses ReverseDiff.jl to compute (∂f/∂y)^T * v efficiently in O(1)
# reverse passes (vs O(n_state) forward passes with ForwardDiff).

"""
    _vjp_state_reversediff!(model, result, state, v, pars, t)

Compute result = (∂f/∂y)^T * v using ReverseDiff.jl.
This is the fallback when symbolic differentiation is not available.
"""
function _vjp_state_reversediff!(model::AbstractOdinModel,
                                  result::AbstractVector{Float64},
                                  state::AbstractVector{Float64},
                                  v::AbstractVector{Float64},
                                  pars, t::Float64)
    n = length(state)
    du = zeros(Float64, n)

    # ReverseDiff computes gradient of scalar function.
    # We want J^T * v = gradient of (v^T * f(y)) w.r.t. y
    function scalar_fn(y)
        du_local = similar(y)
        _odin_rhs!(model, du_local, y, pars, t)
        return dot(v, du_local)
    end

    # Use ReverseDiff tape for efficiency on repeated calls
    result .= ReverseDiff.gradient(scalar_fn, state)
    return nothing
end

"""
    _vjp_params_reversediff!(model, result, state, v, pars, t, param_names)

Compute result = (∂f/∂θ)^T * v using ReverseDiff.jl for specified parameters.
"""
function _vjp_params_reversediff!(model::AbstractOdinModel,
                                   result::AbstractVector{Float64},
                                   state::AbstractVector{Float64},
                                   v::AbstractVector{Float64},
                                   pars, t::Float64,
                                   param_names::Vector{Symbol})
    n_state = length(state)
    n_params = length(param_names)

    # Use ForwardDiff for per-parameter scalar derivatives.
    # ReverseDiff struggles with NamedTuple reconstruction inside _odin_rhs!,
    # but ForwardDiff handles Dual-number propagation cleanly for one param at a time.
    du_buf = Vector{Float64}(undef, n_state)

    for jp in 1:n_params
        pname = param_names[jp]
        pv = Float64(pars[pname])

        g = ForwardDiff.derivative(pv) do theta_j
            pars_mod = merge(pars, NamedTuple{(pname,)}((theta_j,)))
            if model.has_interpolation
                pars_mod = _odin_setup_pars(model, pars_mod)
            end
            T = typeof(theta_j)
            state_T = T.(state)
            du_local = Vector{T}(undef, n_state)
            _odin_rhs!(model, du_local, state_T, pars_mod, t)
            return sum(v[i] * du_local[i] for i in 1:n_state)
        end
        result[jp] = g
    end
    return nothing
end

"""
    _vjp_state_forwarddiff!(model, result, state, v, pars, t)

Compute result = (∂f/∂y)^T * v using ForwardDiff.jl.
Used for validation of symbolic and ReverseDiff results.
"""
function _vjp_state_forwarddiff!(model::AbstractOdinModel,
                                  result::AbstractVector{Float64},
                                  state::AbstractVector{Float64},
                                  v::AbstractVector{Float64},
                                  pars, t::Float64)
    function scalar_fn(y)
        du_local = similar(y)
        _odin_rhs!(model, du_local, y, pars, t)
        return dot(v, du_local)
    end

    result .= ForwardDiff.gradient(scalar_fn, state)
    return nothing
end

"""
    compute_vjp_state!(model, result, state, v, pars, t)

Compute (∂f/∂y)^T * v using the best available method:
1. Symbolic (if available) — fastest, exact
2. ReverseDiff — AD fallback, O(1) reverse passes
3. ForwardDiff — last resort

Returns the method used as a Symbol: :symbolic, :reversediff, or :forwarddiff.
"""
function compute_vjp_state!(model::AbstractOdinModel,
                             result::AbstractVector,
                             state::AbstractVector,
                             v::AbstractVector,
                             pars, t)
    if _odin_has_symbolic_jacobian(model)
        _odin_vjp_state!(model, result, state, v, pars, t)
        return :symbolic
    else
        _vjp_state_reversediff!(model, result, Float64.(state), Float64.(v), pars, Float64(t))
        return :reversediff
    end
end

"""
    compute_vjp_params!(model, result, state, v, pars, t, param_names)

Compute (∂f/∂θ)^T * v using the best available method.
"""
function compute_vjp_params!(model::AbstractOdinModel,
                              result::AbstractVector,
                              state::AbstractVector,
                              v::AbstractVector,
                              pars, t,
                              param_names::Vector{Symbol})
    if _odin_has_symbolic_jacobian(model) && param_names == _odin_diff_param_names(model)
        _odin_vjp_params!(model, result, state, v, pars, t)
        return :symbolic
    else
        _vjp_params_reversediff!(model, result, Float64.(state), Float64.(v),
                                  pars, Float64(t), param_names)
        return :reversediff
    end
end
