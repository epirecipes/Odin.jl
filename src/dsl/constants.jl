# Constants and built-in function/distribution mappings for the odin DSL.

"""Reserved index variable names used in array iterations."""
const INDEX_VARIABLES = (:i, :j, :k, :l, :i5, :i6, :i7, :i8)

"""Reserved names that cannot be used as user variables."""
const RESERVED_NAMES = Set([
    :time, :dt, :i, :j, :k, :l, :i5, :i6, :i7, :i8,
    :initial, :update, :deriv, :diffusion, :dim, :config, :output,
    :parameter, :data, :interpolate, :delay,
    :state, :state_next, :state_deriv, :shared, :internal,
])

"""Special LHS function names in assignments."""
const SPECIAL_LHS = Set([:initial, :update, :deriv, :diffusion, :dim, :config, :output])

"""Special RHS function names in assignments."""
const SPECIAL_RHS = Set([:parameter, :data, :interpolate, :delay])

"""
Mapping from odin distribution names to Distributions.jl constructors.
Each entry maps `OdinName => (JuliaConstructor, n_args, samplable)`.
"""
const DISTRIBUTION_MAP = Dict{Symbol, NamedTuple{(:dist, :nargs, :samplable), Tuple{Any, Int, Bool}}}(
    :Normal       => (dist=Distributions.Normal,       nargs=2, samplable=true),
    :Poisson      => (dist=Distributions.Poisson,      nargs=1, samplable=true),
    :Binomial     => (dist=Distributions.Binomial,     nargs=2, samplable=true),
    :Beta         => (dist=Distributions.Beta,          nargs=2, samplable=true),
    :Gamma        => (dist=Distributions.Gamma,         nargs=2, samplable=true),
    :Uniform      => (dist=Distributions.Uniform,       nargs=2, samplable=true),
    :Exponential  => (dist=Distributions.Exponential,   nargs=1, samplable=true),
    :NegativeBinomial => (dist=Distributions.NegativeBinomial, nargs=2, samplable=true),
    :NegBinomial      => (dist=Distributions.NegativeBinomial, nargs=2, samplable=true),
    :Cauchy       => (dist=Distributions.Cauchy,        nargs=2, samplable=true),
    :LogNormal    => (dist=Distributions.LogNormal,     nargs=2, samplable=true),
    :Weibull      => (dist=Distributions.Weibull,       nargs=2, samplable=true),
    :Hypergeometric => (dist=Distributions.Hypergeometric, nargs=3, samplable=true),
    :BetaBinomial => (dist=Distributions.BetaBinomial,  nargs=3, samplable=true),
    :Multinomial  => (dist=Distributions.Multinomial,   nargs=2, samplable=true),
    # Zero-inflated distributions (3 args: original params + zero-inflation probability)
    :ZIPoisson          => (dist=nothing, nargs=2, samplable=false),
    :ZINegativeBinomial => (dist=nothing, nargs=3, samplable=false),
    :ZINegBinomial      => (dist=nothing, nargs=3, samplable=false),
    # Truncated normal (4 args: mu, sigma, lower, upper)
    :TruncatedNormal    => (dist=nothing, nargs=4, samplable=false),
)

"""
Supported mathematical functions and their Julia equivalents.
Functions not in this list will raise a parse error.
"""
const MATH_FUNCTIONS = Set([
    :+, :-, :*, :/, :^, :%, :÷,
    :cos, :sin, :tan, :acos, :asin, :atan, :atan2,
    :cosh, :sinh, :tanh, :acosh, :asinh, :atanh,
    :exp, :expm1, :log, :log2, :log10, :log1p,
    :sqrt, :abs, :sign, :floor, :ceil, :round, :trunc,
    :gamma, :lgamma, :factorial, :lfactorial, :lbeta, :beta,
    :min, :max, :sum, :prod,
    :>, :<, :>=, :<=, :(==), :(!=),
    :&&, :||, :!,
    :ifelse,
    :length,
])

"""
Reduction functions that operate over arrays.
"""
const REDUCTION_FUNCTIONS = Set([:sum, :prod, :min, :max])
