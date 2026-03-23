# Phase/stage organisation for odin models.
# Determines which equations belong to which evaluation phase.

"""Evaluation phases of a dust system."""
@enum Phase begin
    PHASE_CREATE        # constant-time setup
    PHASE_INITIAL       # compute initial conditions
    PHASE_DYNAMIC       # deriv/update time-stepping
    PHASE_OUTPUT        # extract outputs
    PHASE_COMPARE       # data likelihood
end

"""
Organised model equations, grouped by phase and in dependency order.
"""
struct ModelPhases
    create_eqs::Vector{OdinExpr}       # constant parameter computations
    initial_eqs::Vector{OdinExpr}      # initial condition computations
    dynamic_eqs::Vector{OdinExpr}      # deriv/update equations + their intermediates
    output_eqs::Vector{OdinExpr}       # output extractions
    compare_eqs::Vector{OdinExpr}      # data likelihood computations
    diffusion_eqs::Vector{OdinExpr}    # diffusion coefficient equations
    sorted_intermediates::Vector{Symbol}
end

"""
    organise_phases(exprs, classification, dep_entries)

Assign each equation to a phase and sort within phases.
"""
function organise_phases(
    exprs::Vector{OdinExpr},
    classification::ModelClassification,
    dep_entries::Vector{DepEntry},
)
    # Available symbols from the start: state vars, parameters, data
    available = Set{Symbol}()
    union!(available, classification.state_vars)
    union!(available, keys(classification.parameters))
    union!(available, classification.data_vars)
    union!(available, keys(classification.interpolated))
    push!(available, :time)
    push!(available, :dt)

    # Sort intermediates
    sorted_intermediates = topological_sort(dep_entries, available)

    # Build lookup from name to exprs
    expr_by_name = Dict{Symbol, Vector{OdinExpr}}()
    for ex in exprs
        v = get!(Vector{OdinExpr}, expr_by_name, ex.name)
        push!(v, ex)
    end

    # Separate equations by phase
    create_eqs = OdinExpr[]
    initial_eqs = OdinExpr[]
    dynamic_eqs = OdinExpr[]
    output_eqs = OdinExpr[]
    compare_eqs = OdinExpr[]

    # Parameter defaults go to create
    for ex in exprs
        if ex.type == EXPR_PARAMETER
            push!(create_eqs, ex)
        elseif ex.type == EXPR_DATA
            push!(create_eqs, ex)
        elseif ex.type == EXPR_DIM
            push!(create_eqs, ex)
        elseif ex.type == EXPR_INTERPOLATE
            push!(create_eqs, ex)
        end
    end

    # Initial phase
    for ex in exprs
        if ex.type == EXPR_INITIAL
            push!(initial_eqs, ex)
        end
    end

    # Dynamic phase: sorted intermediates + deriv/update equations
    # Add intermediates in sorted order
    intermediate_set = Set{Symbol}(classification.intermediates)
    intermediates_added = Set{Symbol}()
    for name in sorted_intermediates
        if haskey(expr_by_name, name)
            for ex in expr_by_name[name]
                if ex.type == EXPR_ASSIGNMENT
                    push!(dynamic_eqs, ex)
                    push!(intermediates_added, name)
                end
            end
        end
    end

    # Add deriv/update equations
    for ex in exprs
        if ex.type == EXPR_DERIV || ex.type == EXPR_UPDATE
            push!(dynamic_eqs, ex)
        end
    end

    # Output phase
    for ex in exprs
        if ex.type == EXPR_OUTPUT
            push!(output_eqs, ex)
        end
    end

    # Compare phase
    for ex in exprs
        if ex.type == EXPR_COMPARE
            push!(compare_eqs, ex)
        end
    end

    # Diffusion phase: collect diffusion equations and their intermediate deps
    diffusion_eqs = OdinExpr[]
    diffusion_intermediate_names = Set{Symbol}()

    # Collect intermediate dependencies for diffusion expressions
    for ex in exprs
        if ex.type == EXPR_DIFFUSION
            deps = find_dependencies(ex.rhs)
            setdiff!(deps, INDEX_VARIABLES)
            delete!(deps, ex.name)
            for d in deps
                if d in intermediate_set
                    push!(diffusion_intermediate_names, d)
                end
            end
        end
    end

    # Transitive closure for diffusion intermediates
    changed = true
    while changed
        changed = false
        for ex in exprs
            ex.type == EXPR_ASSIGNMENT || continue
            if ex.name in diffusion_intermediate_names
                deps = find_dependencies(ex.rhs)
                for d in deps
                    if d in intermediate_set && !(d in diffusion_intermediate_names)
                        push!(diffusion_intermediate_names, d)
                        changed = true
                    end
                end
            end
        end
    end

    # Add intermediates needed by diffusion in sorted order
    for name in sorted_intermediates
        if name in diffusion_intermediate_names && haskey(expr_by_name, name)
            for ex in expr_by_name[name]
                if ex.type == EXPR_ASSIGNMENT
                    push!(diffusion_eqs, ex)
                end
            end
        end
    end

    # Add diffusion equations
    for ex in exprs
        if ex.type == EXPR_DIFFUSION
            push!(diffusion_eqs, ex)
        end
    end

    return ModelPhases(
        create_eqs,
        initial_eqs,
        dynamic_eqs,
        output_eqs,
        compare_eqs,
        diffusion_eqs,
        sorted_intermediates,
    )
end
