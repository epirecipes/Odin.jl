# DSL Reference

The `@odin` macro provides a domain-specific language for defining dynamical systems. It supports both continuous (ODE) and discrete-time (stochastic) models.

## Basic Syntax

```julia
gen = @odin begin
    # Equations go here
end
```

The macro returns a `OdinModel` that can create runnable `DustSystem` instances.

## Continuous Models (ODEs)

### `deriv(X) = expression`

Define a derivative for state variable `X`:

```julia
deriv(S) = -beta * S * I / N
```

### `initial(X) = expression`

Set the initial condition for `X`:

```julia
initial(S) = N - I0
```

### `output(X) = expression`

Define an output variable computed during ODE integration but not part of the state:

```julia
output(incidence) = beta * S * I / N
```

## Discrete-Time Models

### `update(X) = expression`

Define the next value of `X` at each time step:

```julia
update(S) = S - n_SI
```

Use `dt` to access the time step size in expressions.

## Parameters and Data

### `X = parameter(default)` / `X = parameter()`

Declare a model parameter:

```julia
beta = parameter(0.5)    # with default
gamma = parameter()       # no default (must be supplied)
```

### `X = data()`

Declare a data variable for use in comparison expressions:

```julia
cases = data()
```

## Stochastic Distributions

Within `update` models, draw random variates:

| Distribution | Syntax |
|-------------|--------|
| Binomial | `Binomial(n, p)` |
| Poisson | `Poisson(lambda)` |
| Normal | `Normal(mean, sd)` |
| NegBinomial | `NegBinomial(size, prob)` |
| Exponential | `Exponential(rate)` |
| Uniform | `Uniform(min, max)` |
| Multinomial | `Multinomial(n, probs)` |
| Categorical | `Categorical(probs)` |

Example:

```julia
n_SI = Binomial(S, p_SI)
```

## Data Comparison

### `X ~ Distribution(args...)`

Compare observed data to model predictions:

```julia
cases ~ Poisson(I)
deaths ~ NegBinomial(size=phi, mu=D)
```

This adds the log-likelihood of observing the data given the model state.

## Arrays

### `dim(X) = n` / `dim(X) = c(n1, n2)`

Declare array dimensions:

```julia
dim(S) = n_age         # 1D array
dim(Y) = c(n_age, 2)  # 2D array
```

### Indexed access

Use bracket notation with index variables `i`, `j`, `k`, `l`:

```julia
deriv(S[i]) = -beta * S[i] * I[i] / N[i]
update(S[i, j]) = S[i, j] - new_infections[i, j]
```

### Array parameters

```julia
dim(contact_rate) = n_age
contact_rate = parameter()
```

### Reductions

| Function | Description |
|----------|-------------|
| `sum(X)` | Sum all elements |
| `sum(X[i, ])` | Sum over second dimension |
| `min(X)` | Minimum |
| `max(X)` | Maximum |

## Interpolation

### `interpolate(times, values, method)`

Create time-varying parameters:

```julia
dim(beta_t) = n_beta
dim(beta_v) = n_beta
beta_t = parameter()
beta_v = parameter()
beta = interpolate(beta_t, beta_v, "constant")   # step function
beta = interpolate(beta_t, beta_v, "linear")      # linear interpolation
beta = interpolate(beta_t, beta_v, "spline")       # cubic spline
```

## Special Variables

| Variable | Description |
|----------|-------------|
| `time` | Current simulation time |
| `dt` | Time step size (discrete models) |
| `step` | Current step number |
| `i`, `j`, `k`, `l` | Array index variables |

## Incidence Tracking

### `zero_every = n`

Reset a variable to zero every `n` time steps (useful for incidence counters):

```julia
update(cases_inc) = cases_inc + new_cases
initial(cases_inc) = 0
# In the model metadata, cases_inc is zeroed at comparison times
```

## Complete Example

```julia
sir_age = @odin begin
    n_age = parameter(3)
    dim(S) = n_age
    dim(I) = n_age
    dim(R) = n_age
    dim(N0) = n_age
    dim(contact) = c(n_age, n_age)

    N0 = parameter()
    contact = parameter()
    gamma = parameter(0.1)

    N_total = sum(S) + sum(I) + sum(R)
    lambda[i] = sum(contact[i, j] * I[j]) / N_total

    update(S[i]) = S[i] - Binomial(S[i], 1 - exp(-lambda[i] * dt))
    update(I[i]) = I[i] + Binomial(S[i], 1 - exp(-lambda[i] * dt)) - Binomial(I[i], 1 - exp(-gamma * dt))
    update(R[i]) = R[i] + Binomial(I[i], 1 - exp(-gamma * dt))

    initial(S[i]) = N0[i] - 1
    initial(I[i]) = 1
    initial(R[i]) = 0
end
```

## DSL Functions

```@docs
Odin.@odin
Odin.@odin_model
```
