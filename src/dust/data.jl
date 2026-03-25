# Data preparation for dust particle filters.

"""
    FilterData{D<:NamedTuple}

Prepared data for a particle filter run, sorted by time.
Parametric on the data element type `D` for type-stable access.
"""
struct FilterData{D<:NamedTuple}
    times::Vector{Float64}
    data::Vector{D}
end

"""
    dust_filter_data(data::AbstractVector{<:NamedTuple}, time_field::Symbol=:time)

Prepare data for a particle filter. Each element must have a `time` field.
Returns sorted `FilterData`.
"""
function dust_filter_data(data::AbstractVector{<:NamedTuple}; time_field::Symbol=:time)
    times = Float64[getfield(d, time_field) for d in data]
    perm = sortperm(times)
    sorted_times = times[perm]
    sorted_data = data[perm]

    # Strip time field from data for passing to compare — preserve concrete NamedTuple type
    first_d = sorted_data[1]
    pairs_list = [k => v for (k, v) in Base.pairs(first_d) if k != time_field]
    example = NamedTuple(pairs_list)
    D = typeof(example)

    clean_data = D[]
    for d in sorted_data
        pairs_list = [k => v for (k, v) in Base.pairs(d) if k != time_field]
        push!(clean_data, NamedTuple(pairs_list)::D)
    end

    return FilterData{D}(sorted_times, clean_data)
end
