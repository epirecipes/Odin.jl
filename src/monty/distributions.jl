# Distribution helpers for the monty DSL.
# Wraps Distributions.jl for log-density and sampling.

"""
    monty_log_density(dist, x) -> Float64

Compute log-density for a Distributions.jl distribution.
"""
monty_log_density(d::Distribution, x) = logpdf(d, x)

"""
    monty_sample(dist, rng) -> Float64

Sample from a Distributions.jl distribution.
"""
monty_sample_dist(d::Distribution, rng::AbstractRNG) = rand(rng, d)
