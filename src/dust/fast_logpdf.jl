# Fast inline logpdf functions that avoid Distributions.jl struct allocation.
# Used by generated compare_data code in the particle filter hot loop.

# Use loggamma from SpecialFunctions (re-exported by Distributions)
const _loggamma = Distributions.SpecialFunctions.loggamma

"""Poisson log-pdf: k*log(λ) - λ - _loggamma(k+1)"""
@inline function _logpdf_poisson(lambda::Real, k::Real)
    k_int = round(Int, k)
    k_int < 0 && return -Inf
    lambda <= 0 && return (k_int == 0 ? 0.0 : -Inf)
    return k_int * log(lambda) - lambda - _loggamma(k_int + 1)
end

"""Normal log-pdf: -0.5*((x-μ)/σ)^2 - log(σ) - 0.5*log(2π)"""
@inline function _logpdf_normal(mu::Real, sigma::Real, x::Real)
    sigma <= 0 && return -Inf
    z = (x - mu) / sigma
    return -0.5 * z * z - log(sigma) - 0.9189385332046727  # 0.5*log(2π)
end

"""NegativeBinomial log-pdf (size r, prob p parameterisation)"""
@inline function _logpdf_negbinomial(r::Real, p::Real, k::Real)
    k_int = round(Int, k)
    k_int < 0 && return -Inf
    return _loggamma(k_int + r) - _loggamma(r) - _loggamma(k_int + 1) +
           r * log(p) + k_int * log(1 - p)
end

"""Binomial log-pdf"""
@inline function _logpdf_binomial(n::Real, p::Real, k::Real)
    k_int = round(Int, k)
    n_int = round(Int, n)
    (k_int < 0 || k_int > n_int) && return -Inf
    p <= 0 && return (k_int == 0 ? 0.0 : -Inf)
    p >= 1 && return (k_int == n_int ? 0.0 : -Inf)
    return _loggamma(n_int + 1) - _loggamma(k_int + 1) - _loggamma(n_int - k_int + 1) +
           k_int * log(p) + (n_int - k_int) * log(1 - p)
end

"""Gamma log-pdf (shape α, scale θ)"""
@inline function _logpdf_gamma(alpha::Real, theta::Real, x::Real)
    x <= 0 && return -Inf
    return (alpha - 1) * log(x) - x / theta - alpha * log(theta) - _loggamma(alpha)
end

"""Exponential log-pdf (scale θ)"""
@inline function _logpdf_exponential(theta::Real, x::Real)
    x < 0 && return -Inf
    return -x / theta - log(theta)
end

"""Beta log-pdf"""
@inline function _logpdf_beta(a::Real, b::Real, x::Real)
    (x <= 0 || x >= 1) && return -Inf
    return (a - 1) * log(x) + (b - 1) * log(1 - x) - (_loggamma(a) + _loggamma(b) - _loggamma(a + b))
end

"""Uniform log-pdf"""
@inline function _logpdf_uniform(a::Real, b::Real, x::Real)
    (x < a || x > b) && return -Inf
    return -log(b - a)
end
