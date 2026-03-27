# Model validation and code introspection utilities.

"""
    OdinValidationResult

Result of validating an odin DSL expression without compiling.
"""
struct OdinValidationResult
    success::Bool
    error::Union{Nothing, String}
    time_type::Union{Nothing, Symbol}   # :continuous or :discrete
    state_variables::Vector{Symbol}
    parameters::Vector{Symbol}
    data_variables::Vector{Symbol}
    intermediates::Vector{Symbol}
    has_compare::Bool
    has_output::Bool
    has_diffusion::Bool
end

"""
    odin_validate(block::Expr) -> OdinValidationResult

Parse an odin DSL expression and return structured diagnostics without compiling.
"""
function odin_validate(block::Expr)
    try
        exprs = parse_odin_block(block)
        cl = classify_variables(exprs)
        dep_entries = build_dependency_graph(exprs, cl)
        phases = organise_phases(exprs, cl, dep_entries)

        return OdinValidationResult(
            true, nothing,
            cl.time_type == TIME_CONTINUOUS ? :continuous : :discrete,
            cl.state_vars,
            collect(keys(cl.parameters)),
            cl.data_vars,
            cl.intermediates,
            !isempty(cl.comparisons),
            !isempty(cl.outputs),
            !isempty(cl.diffusion_vars),
        )
    catch e
        return OdinValidationResult(
            false, sprint(showerror, e),
            nothing, Symbol[], Symbol[], Symbol[], Symbol[],
            false, false, false,
        )
    end
end

"""
    odin_show(block::Expr; what::Symbol=:all) -> Expr

Generate Julia code from an odin DSL block and return it as an expression.

# Arguments
- `block`: An odin DSL block expression.
- `what`: Which part to return. `:all` for the full generated code, or a
  specific method name (`:update`, `:rhs`, `:initial`, `:compare`, `:output`,
  `:diffusion`) to filter to that function.
"""
function odin_show(block::Expr; what::Symbol=:all)
    exprs = parse_odin_block(block)
    cl = classify_variables(exprs)
    dep_entries = build_dependency_graph(exprs, cl)
    phases = organise_phases(exprs, cl, dep_entries)
    code = generate_system(exprs, cl, phases)

    if what == :all
        return code
    else
        return _filter_generated_code(code, what)
    end
end

"""
    _filter_generated_code(code::Expr, what::Symbol) -> Union{Expr, Nothing}

Extract a specific function definition from generated code by matching method name.
"""
function _filter_generated_code(code::Expr, what::Symbol)
    target_bang = "_odin_$(what)!"
    target_nobang = "_odin_$(what)"

    matches = Expr[]
    _find_function_defs!(matches, code, target_bang, target_nobang)

    if length(matches) == 1
        return matches[1]
    elseif length(matches) > 1
        return Expr(:block, matches...)
    else
        return nothing
    end
end

function _find_function_defs!(matches::Vector{Expr}, expr::Expr, target_bang::String, target_nobang::String)
    if expr.head == :function
        fname = _extract_function_name(expr)
        if fname !== nothing
            s = string(fname)
            if s == target_bang || s == target_nobang
                push!(matches, expr)
                return
            end
        end
    end
    for arg in expr.args
        if arg isa Expr
            _find_function_defs!(matches, arg, target_bang, target_nobang)
        end
    end
end

function _extract_function_name(fexpr::Expr)
    if fexpr.head == :function && length(fexpr.args) >= 1
        sig = fexpr.args[1]
        if sig isa Expr && sig.head == :call
            name_part = sig.args[1]
            # Handle qualified names like Odin._odin_rhs!
            if name_part isa Expr && name_part.head == :(.)
                return _dot_last(name_part)
            elseif name_part isa Symbol
                return name_part
            end
        elseif sig isa Expr && sig.head == :where
            # function f(args...) where T
            inner = sig.args[1]
            if inner isa Expr && inner.head == :call
                name_part = inner.args[1]
                if name_part isa Expr && name_part.head == :(.)
                    return _dot_last(name_part)
                elseif name_part isa Symbol
                    return name_part
                end
            end
        end
    end
    return nothing
end

function _dot_last(expr::Expr)
    # expr is like :(Odin._odin_rhs!) => args = [:Odin, QuoteNode(:_odin_rhs!)]
    last = expr.args[end]
    if last isa QuoteNode
        return last.value
    elseif last isa Symbol
        return last
    end
    return nothing
end
