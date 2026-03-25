# Fast inline random samplers that avoid Distributions.jl constructor overhead.
# These are called from generated model code in the inner simulation loop.

using PoissonRandom: pois_rand

"""
    _rand_poisson(rng, λ) -> Int

Fast Poisson draw using PoissonRandom.jl's `pois_rand`.
"""
@inline _rand_poisson(rng::AbstractRNG, λ::Real) = pois_rand(rng, Float64(λ))

"""
    _rand_binomial(rng, n, p) -> Int

Fast inline binomial sampler. Uses BINV (inverse CDF / geometric) for small n*p,
BTPE (Kachitvichyanukul & Schmeiser 1988) for large n*p.
Avoids creating a Distributions.Binomial struct per call.
"""
@inline function _rand_binomial(rng::AbstractRNG, n::Int, p::Float64)
    n <= 0 && return 0
    p <= 0.0 && return 0
    p >= 1.0 && return n

    if p > 0.5
        return n - _rand_binomial(rng, n, 1.0 - p)
    end

    q = 1.0 - p
    np = n * p

    if np < 10.0
        return _binomial_binv(rng, n, p, q, np)
    else
        return _binomial_btpe(rng, n, p, q, np)
    end
end

# BINV: inverse CDF method for small n*p
@inline function _binomial_binv(rng::AbstractRNG, n::Int, p::Float64, q::Float64, np::Float64)
    s = p / q
    a = (n + 1) * s
    r = q ^ n
    u = rand(rng)
    x = 0
    while u > r
        u -= r
        x += 1
        r *= (a / x - s)
    end
    return x
end

# BTPE: triangle-parallelogram-exponential method for large n*p
@inline function _binomial_btpe(rng::AbstractRNG, n::Int, p::Float64, q::Float64, np::Float64)
    fm = np + p
    m = floor(Int, fm)
    p1 = floor(2.195 * sqrt(np * q) - 4.6 * q) + 0.5
    xm = m + 0.5
    xl = xm - p1
    xr = xm + p1
    c = 0.134 + 20.5 / (15.3 + m)
    al = (fm - xl) / (fm - xl * p)
    λ_l = al * (1.0 + 0.5 * al)
    ar = (xr - fm) / (xr * q)
    λ_r = ar * (1.0 + 0.5 * ar)
    p2 = p1 * (1.0 + 2.0 * c)
    p3 = p2 + c / λ_l
    p4 = p3 + c / λ_r

    while true
        u = rand(rng) * p4
        v = rand(rng)

        local y::Int
        if u <= p1
            y = floor(Int, xm - p1 * v + u)
            return y
        elseif u <= p2
            x_val = xl + (u - p1) / c
            v = v * c + 1.0 - abs(m - x_val + 0.5) / p1
            v > 1.0 && continue
            y = floor(Int, x_val)
        elseif u <= p3
            y = floor(Int, xl + log(v) / λ_l)
            y < 0 && continue
            v *= (u - p2) * λ_l
        else
            y = floor(Int, xr - log(v) / λ_r)
            y > n && continue
            v *= (u - p3) * λ_r
        end

        k = abs(y - m)
        if k <= 20 || k >= 0.5 * np * q - 1.0
            s_val = p / q
            a_val = (n + 1) * s_val
            F = 1.0
            if m < y
                for i in (m+1):y
                    F *= (a_val / i - s_val)
                end
            elseif m > y
                for i in (y+1):m
                    F /= (a_val / i - s_val)
                end
            end
            v <= F && return y
        else
            rho = (k / (np * q)) * ((k * (k / 3.0 + 0.625) + 1.0 / 6.0) / (np * q) + 0.5)
            t = -k * k / (2.0 * np * q)
            A = log(v)
            A < t - rho && return y
            A > t + rho && continue
            # Full Stirling comparison
            x1 = y + 1.0; f1 = m + 1.0; z = n + 1.0 - m; w = n - y + 1.0
            x2 = x1 * x1; f2 = f1 * f1; z2 = z * z; w2 = w * w
            bound = xm * log(f1/x1) + (n-m+0.5)*log(z/w) + (y-m)*log(w*p/(x1*q)) +
                (13860.0-(462.0-(132.0-(99.0-140.0/f2)/f2)/f2)/f2)/f1/166320.0 +
                (13860.0-(462.0-(132.0-(99.0-140.0/z2)/z2)/z2)/z2)/z/166320.0 -
                (13860.0-(462.0-(132.0-(99.0-140.0/x2)/x2)/x2)/x2)/x1/166320.0 -
                (13860.0-(462.0-(132.0-(99.0-140.0/w2)/w2)/w2)/w2)/w/166320.0
            A > bound && continue
            return y
        end
    end
end

"""
    _rand_normal(rng, μ, σ) -> Float64

Inline normal draw — just calls randn (already very fast).
"""
@inline _rand_normal(rng::AbstractRNG, μ::Real, σ::Real) = μ + σ * randn(rng)

"""
    _rand_nbinom(rng, r, p) -> Int

Negative binomial via Gamma-Poisson mixture. Works for real-valued r (size).
"""
@inline function _rand_nbinom(rng::AbstractRNG, r::Int, p::Float64)
    r <= 0 && return 0
    p <= 0.0 && return 0
    p >= 1.0 && return 0
    # Gamma(r, (1-p)/p) then Poisson
    λ = rand(rng, Distributions.Gamma(Float64(r), (1.0 - p) / p))
    return pois_rand(rng, λ)
end

"""
    _rand_exponential(rng, rate) -> Float64

Inline exponential draw with rate parameterisation (mean = 1/rate).
"""
@inline _rand_exponential(rng::AbstractRNG, rate::Real) = randexp(rng) / rate

"""
    _rand_gamma(rng, shape, rate) -> Float64

Gamma draw. Falls back to Distributions.jl (fast enough, rarely in inner loops).
"""
@inline _rand_gamma(rng::AbstractRNG, shape::Real, rate::Real) =
    rand(rng, Distributions.Gamma(shape, 1.0 / rate))

"""
    _rand_multinomial!(rng, out, n, prob)

In-place multinomial draw via sequential conditional binomials.
`out` is filled with the counts; `prob` is a vector of probabilities (need not sum to 1).
This is the standard decomposition used by odin2 for multinomial draws.
"""
function _rand_multinomial!(rng::AbstractRNG, out::AbstractVector{Int},
                            n::Int, prob::AbstractVector{<:Real})
    k = length(prob)
    @assert length(out) >= k
    remaining = n
    prob_remaining = sum(prob)
    @inbounds for i in 1:(k-1)
        if remaining <= 0 || prob_remaining <= 0.0
            out[i] = 0
        else
            p_i = prob[i] / prob_remaining
            out[i] = _rand_binomial(rng, remaining, clamp(p_i, 0.0, 1.0))
            remaining -= out[i]
            prob_remaining -= prob[i]
        end
    end
    @inbounds out[k] = remaining
    return out
end

"""
    _rand_multinomial(rng, n, prob) -> Vector{Int}

Allocating multinomial draw (convenience wrapper).
"""
function _rand_multinomial(rng::AbstractRNG, n::Int, prob::AbstractVector{<:Real})
    out = Vector{Int}(undef, length(prob))
    _rand_multinomial!(rng, out, n, prob)
    return out
end
