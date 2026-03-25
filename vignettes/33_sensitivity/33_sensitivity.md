# Sensitivity Analysis for ODE Models


## Overview

Sensitivity analysis answers the question: *how does model output change
when parameters change?* Odin.jl provides four complementary approaches:

| Method | Type | Best for | Cost |
|----|----|----|----|
| **Forward** | Local, gradient | Few parameters, full trajectory | O(n_state × n_params) |
| **Adjoint** | Local, gradient | Many parameters, scalar loss | O(n_state) backward |
| **Sobol** | Global, variance-based | Parameter importance ranking | O(N × (k+2)) model runs |
| **Morris** | Global, screening | Cheap screening of many params | O(N × (k+1)) model runs |

## SIR Model

We use a simple SIR model throughout:

``` julia
using Odin

sir = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10)
    N = parameter(1000)
end

pars = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
times = collect(1.0:1.0:100.0)
```

    100-element Vector{Float64}:
       1.0
       2.0
       3.0
       4.0
       5.0
       6.0
       7.0
       8.0
       9.0
      10.0
       ⋮
      92.0
      93.0
      94.0
      95.0
      96.0
      97.0
      98.0
      99.0
     100.0

## Forward Sensitivity

Forward sensitivity solves the *variational equation* alongside the ODE:

$$\frac{dS_{ij}}{dt} = \sum_k \frac{\partial f_i}{\partial u_k} S_{kj} + \frac{\partial f_i}{\partial p_j}$$

where $S_{ij} = \partial u_i / \partial p_j$ is the sensitivity of state
$i$ to parameter $j$.

``` julia
result = dust_sensitivity_forward(sir, pars;
    times=times,
    params_of_interest=[:beta, :gamma])

# result.trajectory — (3, 100) state trajectory
# result.sensitivities — (3, 2, 100) sensitivity array
```

    ForwardSensitivityResult([983.951470083419 975.0700840760265 … 6.913251846251666 6.91248184361251; 14.822856654009431 21.89079721215265 … 0.23370064330922954 0.2121932707273185; 1.2256732625715523 3.0391187118207545 … 992.8530475104393 992.8753248856603], [-15.227261837949492 3.211449684331859; 14.583483342884033 -14.815416565023776; 0.6437780712675876 11.603966807982088;;; -46.03223090409246 16.710518901897725; 42.66863606023852 -43.67415482745682; 3.363592819811587 26.963634685441527;;; -102.58642035725636 48.87067077740479; 92.6817426532303 -95.8122326622592; 9.9046726377027 46.941562262916264;;; … ;;; -71.15128780469361 354.88429923122305; -1.3428648356544324 -17.637596190813035; 72.49561154049132 -337.2411973899261;;; -71.13981919642968 354.8992250800515; -1.2259795171713532 -16.206651654840297; 72.36725762244903 -338.68706770165636;;; -71.12938549805737 354.9134028650859; -1.1192333349423225 -14.889698285868247; 72.25007775750922 -340.0181989132846], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0], [:beta, :gamma])

### Sensitivity of I to β over time

The sensitivity $\partial I / \partial \beta$ shows how the infected
compartment responds to changes in the transmission rate:

``` julia
using Printf

println("Time  |  I(t)   |  ∂I/∂β   |  ∂I/∂γ")
println("------|---------|----------|--------")
for ti in [10, 20, 30, 40, 50]
    I_val = result.trajectory[2, ti]
    dI_dbeta = result.sensitivities[2, 1, ti]
    dI_dgamma = result.sensitivities[2, 2, ti]
    @printf("%4d  | %7.1f | %8.1f | %8.1f\n", ti, I_val, dI_dbeta, dI_dgamma)
end
```

    Time  |  I(t)   |  ∂I/∂β   |  ∂I/∂γ
    ------|---------|----------|--------
      10  |   294.4 |   1647.5 |  -2229.5
      20  |   400.3 |   -452.6 |  -2937.7
      30  |   175.0 |   -493.6 |  -2323.8
      40  |    68.6 |   -234.0 |  -1439.3
      50  |    26.4 |    -99.9 |   -766.5

The sensitivity to β is positive (more transmission → more infected),
while the sensitivity to γ is negative (faster recovery → fewer
infected), as expected.

## Adjoint Sensitivity

When you need the gradient of a **scalar** loss function (e.g.,
log-likelihood), the adjoint method is more efficient than forward
sensitivity for many parameters.

``` julia
# Define a simple squared-error loss at observed times
obs_times = collect(10.0:10.0:50.0)
obs_I = [50.0, 200.0, 400.0, 300.0, 150.0]  # synthetic observations

loss_fn = (state, t) -> begin
    I_pred = state[2]
    idx = findfirst(τ -> abs(τ - t) < 0.5, obs_times)
    idx === nothing && return 0.0
    return -0.5 * (I_pred - obs_I[idx])^2 / 100.0  # Gaussian log-likelihood
end

adj = dust_sensitivity_adjoint(sir, pars, loss_fn;
    times=obs_times,
    params_of_interest=[:beta, :gamma])

println("Loss value: ", round(adj.loss_value, digits=2))
println("∂L/∂β = ", round(adj.gradient[1], digits=4))
println("∂L/∂γ = ", round(adj.gradient[2], digits=4))
```

    Loss value: -1096.47
    ∂L/∂β = -1205.8045
    ∂L/∂γ = -4524.644

## Integration with the Unfilter

For models with a `compare` block and observed data, the gradient of the
log-likelihood can be computed directly:

``` julia
sir_compare = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0
    obs = data()
    obs ~ Poisson(max(I, 1e-6))
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10)
    N = parameter(1000)
end

# Generate synthetic data
sys = dust_system_create(sir_compare, pars)
dust_system_set_state_initial!(sys)
sim = dust_system_simulate(sys, collect(5.0:5.0:50.0))
data_vec = [(time=5.0*i, obs=max(1.0, round(sim[2,1,i]))) for i in 1:10]
fdata = Odin.dust_filter_data(data_vec)

unfilter = dust_unfilter_create(sir_compare, fdata)
packer = monty_packer([:beta, :gamma]; fixed=(I0=10.0, N=1000.0))

# Forward sensitivity gradient
fwd = dust_unfilter_gradient(unfilter, pars, packer; method=:forward)
println("Forward: LL = $(round(fwd.log_likelihood, digits=2)), ",
        "grad = [$(round(fwd.gradient[1], digits=3)), $(round(fwd.gradient[2], digits=3))]")

# Adjoint gradient
adj = dust_unfilter_gradient(unfilter, pars, packer; method=:adjoint)
println("Adjoint: LL = $(round(adj.log_likelihood, digits=2)), ",
        "grad = [$(round(adj.gradient[1], digits=3)), $(round(adj.gradient[2], digits=3))]")
```

    Forward: LL = -33.6, grad = [-2.766, -5.136]
    Adjoint: LL = -33.6, grad = [-3.227, -4.721]

## Sobol Sensitivity Indices

Sobol indices decompose output variance into contributions from each
parameter. The **first-order index** $S_i$ measures the direct effect of
parameter $i$; the **total-order index** $S_{T_i}$ includes
interactions.

``` julia
pars_ranges = Dict(
    :beta  => (0.2, 1.0),
    :gamma => (0.05, 0.3),
    :I0    => (1.0, 50.0),
    :N     => (500.0, 2000.0),
)

sobol = dust_sensitivity_sobol(sir, pars_ranges;
    n_samples=500,
    times=collect(5.0:5.0:50.0),
    output_var=2)  # I compartment at final time

println("Parameter  | First-order | Total-order")
println("-----------|-------------|------------")
for k in sobol.param_names
    @printf("%-10s | %11.3f | %11.3f\n", k, sobol.first_order[k], sobol.total_order[k])
end
```

    Parameter  | First-order | Total-order
    -----------|-------------|------------
    beta       |       0.166 |       0.344
    gamma      |       0.392 |       0.678
    N          |       0.051 |       0.184
    I0         |       0.030 |       0.066

Parameters with high total-order but low first-order indices have
important **interactions** with other parameters.

## Morris Screening

Morris screening is a computationally cheap way to rank parameter
importance. It computes the mean absolute elementary effect ($\mu^*$)
and its standard deviation ($\sigma$). High $\mu^*$ means the parameter
is influential; high $\sigma$ means its effect depends on other
parameters (interactions or nonlinearity).

``` julia
morris = dust_sensitivity_morris(sir, pars_ranges;
    n_trajectories=30,
    times=collect(5.0:5.0:50.0),
    output_var=:I)

println("Parameter  |    μ*    |    σ")
println("-----------|----------|--------")
for k in morris.param_names
    @printf("%-10s | %8.1f | %8.1f\n", k, morris.mu_star[k], morris.sigma[k])
end
```

    Parameter  |    μ*    |    σ
    -----------|----------|--------
    beta       |     99.8 |    213.2
    gamma      |    223.8 |    255.8
    N          |     95.7 |    178.4
    I0         |     17.6 |     43.7

### Interpretation

- **μ\* ≫ 0**: parameter has important influence on the output
- **σ/μ\* \> 1**: strong interactions or nonlinearity
- **σ/μ\* ≈ 0**: nearly linear, additive effect

## Comparison with R

In R, similar analyses can be performed with the `deSolve` and
`sensitivity` packages:

``` r
library(deSolve)

sir_ode <- function(t, y, pars) {
  with(as.list(c(y, pars)), {
    dS <- -beta * S * I / N
    dI <- beta * S * I / N - gamma * I
    dR <- gamma * I
    list(c(dS, dI, dR))
  })
}

pars <- c(beta = 0.5, gamma = 0.1, N = 1000, I0 = 10)
y0 <- c(S = 990, I = 10, R = 0)
times <- seq(1, 100, by = 1)

# Forward sensitivity using deSolve::sensFun
library(FME)
sf <- sensFun(func = sir_ode, parms = pars, sensvar = "I",
              varscale = 1, parscale = 1,
              y = y0, times = times)

# Sobol via sensitivity package
library(sensitivity)
sobol_result <- sobolSalt(model = NULL, X1, X2, nboot = 100)
```

The odin2 R package provides adjoint sensitivity natively when models
include the `adjoint()` method in the generated C++ code.

## Summary

| Use case | Recommended method |
|----|----|
| Trajectory sensitivity (few params) | `dust_sensitivity_forward` |
| Gradient for optimisation/MCMC | `dust_unfilter_gradient` with `:adjoint` |
| Parameter importance ranking | `dust_sensitivity_sobol` |
| Quick screening (many params) | `dust_sensitivity_morris` |
