# Parameter packing/unpacking: maps between named parameters and flat vectors.

"""
    MontyPacker

Maps between a flat Float64 vector (for samplers) and a NamedTuple (for models).
"""
struct MontyPacker
    names::Vector{Symbol}
    scalar_names::Vector{Symbol}
    array_names::Vector{Symbol}
    array_dims::Dict{Symbol, Tuple}
    index::Dict{Symbol, UnitRange{Int}}
    len::Int
    fixed::NamedTuple
    process::Union{Nothing, Function}
end

"""
    monty_packer(scalar=Symbol[], array=Dict{Symbol,Tuple}(); fixed=NamedTuple(), process=nothing)

Create a parameter packer.

## Example
```julia
p = monty_packer([:beta, :gamma])
p = monty_packer([:beta]; array=Dict(:gamma => (3,)), fixed=(N=1000,))
```
"""
function monty_packer(
    scalar::Vector{Symbol}=Symbol[];
    array::Dict{Symbol, <:Any}=Dict{Symbol, Tuple}(),
    fixed::NamedTuple=NamedTuple(),
    process::Union{Nothing, Function}=nothing,
)
    names = Symbol[]
    index = Dict{Symbol, UnitRange{Int}}()
    array_dims = Dict{Symbol, Tuple}()
    pos = 1

    # Scalars first
    for s in scalar
        push!(names, s)
        index[s] = pos:pos
        pos += 1
    end

    # Arrays
    array_names = Symbol[]
    for (name, dims) in array
        push!(names, name)
        push!(array_names, name)
        n = prod(dims isa Integer ? (dims,) : dims)
        d = dims isa Integer ? (dims,) : Tuple(dims)
        array_dims[name] = d
        index[name] = pos:(pos + n - 1)
        pos += n
    end

    len = pos - 1

    return MontyPacker(names, scalar, array_names, array_dims, index, len, fixed, process)
end

"""
    (packer::MontyPacker)(x::AbstractVector) -> NamedTuple

Unpack a vector into a NamedTuple.
"""
function (packer::MontyPacker)(x::AbstractVector)
    return unpack(packer, x)
end

function unpack(packer::MontyPacker, x::AbstractVector{T}) where {T}
    @assert length(x) == packer.len "Expected vector of length $(packer.len), got $(length(x))"

    # Fast path: scalar-only with no process function (common case: 2-5 params)
    if isempty(packer.array_names) && packer.process === nothing
        # Build NamedTuple directly without intermediate Pair array
        all_names = packer.scalar_names
        n_scalar = length(all_names)
        fixed_keys = keys(packer.fixed)
        n_fixed = length(fixed_keys)
        n_total = n_scalar + n_fixed

        vals = Vector{Any}(undef, n_total)
        names = Vector{Symbol}(undef, n_total)
        @inbounds for i in 1:n_scalar
            names[i] = all_names[i]
            vals[i] = x[first(packer.index[all_names[i]])]
        end
        @inbounds for (j, k) in enumerate(fixed_keys)
            names[n_scalar + j] = k
            vals[n_scalar + j] = packer.fixed[k]
        end
        return NamedTuple{Tuple(names)}(Tuple(vals))
    end

    # General path with arrays and/or process function
    pairs_list = Pair{Symbol, Any}[]

    for name in packer.scalar_names
        r = packer.index[name]
        push!(pairs_list, name => x[first(r)])
    end

    for name in packer.array_names
        r = packer.index[name]
        dims = packer.array_dims[name]
        if length(dims) == 1
            push!(pairs_list, name => x[r])
        else
            push!(pairs_list, name => reshape(x[r], dims))
        end
    end

    # Add fixed values
    for (k, v) in Base.pairs(packer.fixed)
        push!(pairs_list, k => v)
    end

    nt = NamedTuple(pairs_list)

    # Apply process function
    if packer.process !== nothing
        extra = packer.process(nt)
        nt = merge(nt, extra)
    end

    return nt
end

"""
    pack(packer::MontyPacker, nt::NamedTuple) -> Vector{Float64}

Pack a NamedTuple into a flat vector.
"""
function pack(packer::MontyPacker, nt::NamedTuple)
    x = zeros(Float64, packer.len)

    for name in packer.scalar_names
        r = packer.index[name]
        x[first(r)] = Float64(getfield(nt, name))
    end

    for name in packer.array_names
        r = packer.index[name]
        val = getfield(nt, name)
        x[r] .= vec(val)
    end

    return x
end

"""
    parameter_names(packer::MontyPacker) -> Vector{String}

Return the full parameter names including array element names.
"""
function parameter_names(packer::MontyPacker)
    names = String[]
    for s in packer.scalar_names
        push!(names, string(s))
    end
    for name in packer.array_names
        dims = packer.array_dims[name]
        n = prod(dims)
        for i in 1:n
            push!(names, "$(name)[$i]")
        end
    end
    return names
end

# ── Grouped packer ──────────────────────────────────────────

"""
    MontyPackerGrouped

Packer for multiple parameter groups (shared + group-specific parameters).
"""
struct MontyPackerGrouped
    groups::Vector{Symbol}
    shared_names::Vector{Symbol}
    varied_names::Vector{Symbol}
    base_packer::MontyPacker
    len::Int
    n_shared::Int
    n_varied_per_group::Int
end

"""
    monty_packer_grouped(groups, shared, varied; fixed, process)

Create a grouped parameter packer.
"""
function monty_packer_grouped(
    groups::Vector{Symbol};
    shared::Vector{Symbol}=Symbol[],
    varied::Vector{Symbol}=Symbol[],
    fixed::NamedTuple=NamedTuple(),
    process::Union{Nothing, Function}=nothing,
)
    n_groups = length(groups)
    n_shared = length(shared)
    n_varied = length(varied)
    len = n_shared + n_varied * n_groups

    # Create base packer for a single group's full parameter set
    base = monty_packer(vcat(shared, varied); fixed=fixed, process=process)

    return MontyPackerGrouped(groups, shared, varied, base, len, n_shared, n_varied)
end

function unpack(packer::MontyPackerGrouped, x::AbstractVector)
    @assert length(x) == packer.len
    shared_vals = x[1:packer.n_shared]

    result = Dict{Symbol, NamedTuple}()
    offset = packer.n_shared
    for (gi, g) in enumerate(packer.groups)
        varied_vals = x[(offset + 1):(offset + packer.n_varied_per_group)]
        offset += packer.n_varied_per_group

        full_vec = vcat(shared_vals, varied_vals)
        result[g] = unpack(packer.base_packer, full_vec)
    end

    return result
end
