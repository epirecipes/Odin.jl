# Interpolation support for odin models.
# Wraps Interpolations.jl for constant/linear/spline time-varying parameters.

"""
    build_interpolator(times, values, mode::Symbol)

Create an interpolation function from time and value vectors.
Mode is one of :constant, :linear, :spline.
"""
function build_interpolator(times::AbstractVector, values::AbstractVector, mode::Symbol)
    if mode == :constant
        return _constant_interpolator(times, values)
    elseif mode == :linear
        return _linear_interpolator(times, values)
    elseif mode == :spline
        return _spline_interpolator(times, values)
    else
        error("Unknown interpolation mode: $mode")
    end
end

function _constant_interpolator(times::AbstractVector, values::AbstractVector)
    # Piecewise constant: return value at last time point <= t
    function interp(t)
        idx = searchsortedlast(times, t)
        idx = clamp(idx, 1, length(values))
        return @inbounds values[idx]
    end
    return interp
end

function _linear_interpolator(times::AbstractVector, values::AbstractVector)
    itp = LinearInterpolation(times, values)
    function interp(t)
        t_clamped = clamp(t, first(times), last(times))
        return itp(t_clamped)
    end
    return interp
end

function _spline_interpolator(times::AbstractVector, values::AbstractVector)
    itp = CubicSplineInterpolation(times, values)
    function interp(t)
        t_clamped = clamp(t, first(times), last(times))
        return itp(t_clamped)
    end
    return interp
end

# Interpolations.jl convenience wrappers
function LinearInterpolation(xs, ys)
    knots = (xs,)
    itp = Interpolations.interpolate(knots, ys, Gridded(Interpolations.Linear()))
    return itp
end

function CubicSplineInterpolation(xs, ys)
    # cubic_spline_interpolation requires uniform spacing (Range input).
    # If xs is already a range, use directly. Otherwise check if uniformly
    # spaced and convert, or fall back to BSpline interpolation.
    if xs isa AbstractRange
        return Interpolations.cubic_spline_interpolation(xs, ys)
    end
    # Check for uniform spacing
    diffs = diff(xs)
    if all(d -> abs(d - diffs[1]) < 1e-12 * abs(diffs[1]), diffs)
        rng = range(first(xs), last(xs), length=length(xs))
        return Interpolations.cubic_spline_interpolation(rng, ys)
    end
    # Non-uniform knots: use Gridded linear + BSpline fallback
    # (Interpolations.jl doesn't natively support cubic with non-uniform knots)
    knots = (xs,)
    return Interpolations.interpolate(knots, ys, Gridded(Interpolations.Linear()))
end
