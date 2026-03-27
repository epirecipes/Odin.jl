# The @odin macro: main user-facing entry point for model definition.

"""
    AbstractOdinModel

Abstract type for all odin-generated models.
"""
abstract type AbstractOdinModel end

# Method stubs that generated models implement
function _odin_initial! end
function _odin_rhs! end
function _odin_update! end
function _odin_diffusion! end
function _odin_compare_data end
function _odin_zero_every end
function _odin_n_state end
function _odin_state_names end
function _odin_output! end
function _odin_n_output end
function _odin_output_names end
function _odin_setup_pars end
function _odin_delay_tau_values end

# Symbolic differentiation stubs (overridden when symbolic succeeds)
function _odin_vjp_state! end
function _odin_vjp_params! end
function _odin_diff_param_names end
function _odin_n_diff_params end

"""Default: model does not have symbolic Jacobian."""
_odin_has_symbolic_jacobian(::AbstractOdinModel) = false

"""Default: model does not have delay."""
_odin_has_delay(model::AbstractOdinModel) = hasproperty(model, :has_delay) && model.has_delay

"""Default: no differentiated parameters."""
_odin_diff_param_names(::AbstractOdinModel) = Symbol[]

"""Default: zero differentiated parameters."""
_odin_n_diff_params(::AbstractOdinModel) = 0

"""
    @odin(block)

Define an odin model using the DSL. Returns a `DustSystemGenerator`.

## Example

```julia
sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    N = parameter(1000)
    I0 = parameter(10)
    beta = parameter(0.2)
    gamma = parameter(0.1)
end
```
"""
macro odin(block)
    # Parse the block
    exprs = parse_odin_block(block)

    # Classify variables
    classification = classify_variables(exprs)

    # Build dependency graph
    dep_entries = build_dependency_graph(exprs, classification)

    # Organise into phases
    phases = organise_phases(exprs, classification, dep_entries)

    # Generate code
    code = generate_system(exprs, classification, phases)

    return esc(code)
end
