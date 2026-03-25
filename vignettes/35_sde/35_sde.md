# Stochastic Differential Equations (SDE)


## Introduction

Stochastic differential equations (SDEs) extend ordinary differential
equations (ODEs) with continuous noise terms, modelling systems subject
to random fluctuations. In the Itô formulation:

$$dX = f(X, t)\,dt + \sigma(X, t)\,dW$$

where $f$ is the drift (deterministic dynamics), $\sigma$ is the
diffusion coefficient (noise intensity), and $dW$ is a Wiener process
increment.

Odin supports SDEs via the `diffusion()` keyword alongside `deriv()`.
Each state variable can optionally have a diffusion term specifying its
noise coefficient.

## SIR SDE model

We define a stochastic SIR model where the noise scales with the square
root of the transition rates (demographic noise):

``` julia
using Odin
using Statistics

sir_sde = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    diffusion(S) = sigma * sqrt(abs(beta * S * I / N))
    diffusion(I) = sigma * sqrt(abs(beta * S * I / N + gamma * I))
    diffusion(R) = sigma * sqrt(abs(gamma * I))
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0.0
    beta = parameter(0.5)
    gamma = parameter(0.1)
    sigma = parameter(0.1)
    I0 = parameter(10.0)
    N = parameter(1000.0)
end
```

    DustSystemGenerator{var"##OdinModel#277"}(var"##OdinModel#277"(3, [:S, :I, :R], [:beta, :gamma, :sigma, :I0, :N], true, true, false, false, false, Dict{Symbol, Array}()))

The `diffusion(X) = expr` line specifies $\sigma_X(X, t)$ for each state
variable. States without a `diffusion()` line receive zero noise.

## Simulating the SDE model

SDE models use fixed time-stepping. The `dt` parameter in
`dust_system_create` sets the integration step size (smaller = more
accurate but slower):

``` julia
pars = (beta=0.5, gamma=0.1, sigma=0.1, I0=10.0, N=1000.0)
sys = dust_system_create(sir_sde, pars; dt=0.01, seed=42)
dust_system_set_state_initial!(sys)

times = collect(0.0:0.5:80.0)
output = dust_system_simulate(sys, times)
println("Output shape: ", size(output))
```

    Output shape: (3, 1, 161)

## Comparing ODE vs SDE

Let’s compare a single SDE trajectory against the deterministic ODE:

``` julia
sir_ode = @odin begin
    deriv(S) = -beta * S * I / N
    deriv(I) = beta * S * I / N - gamma * I
    deriv(R) = gamma * I
    initial(S) = N - I0
    initial(I) = I0
    initial(R) = 0.0
    beta = parameter(0.5)
    gamma = parameter(0.1)
    I0 = parameter(10.0)
    N = parameter(1000.0)
end

pars_ode = (beta=0.5, gamma=0.1, I0=10.0, N=1000.0)
sys_ode = dust_system_create(sir_ode, pars_ode)
dust_system_set_state_initial!(sys_ode)
out_ode = dust_system_simulate(sys_ode, times; solver=:dp5)

println("ODE I peak: ", maximum(out_ode[2, 1, :]))
println("SDE I peak: ", maximum(output[2, 1, :]))
```

    ODE I peak: 479.9754786185972
    SDE I peak: 474.6538048324087

## Ensemble of SDE paths

Running multiple particles gives an ensemble of stochastic trajectories:

``` julia
n_particles = 100
sys_ensemble = dust_system_create(sir_sde, pars; n_particles=n_particles, dt=0.01, seed=123)
dust_system_set_state_initial!(sys_ensemble)
ensemble = dust_system_simulate(sys_ensemble, times)

# Compute mean and standard deviation across particles
I_mean = vec(mean(ensemble[2, :, :], dims=1))
I_std = vec(std(ensemble[2, :, :], dims=1))

println("Mean peak I: ", maximum(I_mean))
println("Std at peak: ", I_std[argmax(I_mean)])
```

    Mean peak I: 480.12884384971454
    Std at peak: 3.589692220888349

## Euler-Maruyama vs Milstein

Odin provides two SDE solvers:

- **Euler-Maruyama** (`:euler_maruyama`): Strong order 0.5, simplest
  scheme
- **Milstein** (`:milstein`): Strong order 1.0, uses derivative of
  diffusion coefficient

``` julia
# Euler-Maruyama
sys_em = dust_system_create(sir_sde, pars; dt=0.01, seed=42)
dust_system_set_state_initial!(sys_em)
out_em = dust_system_simulate(sys_em, times; solver=:euler_maruyama)

# Milstein
sys_mil = dust_system_create(sir_sde, pars; dt=0.01, seed=42)
dust_system_set_state_initial!(sys_mil)
out_mil = dust_system_simulate(sys_mil, times; solver=:milstein)

println("EM final I:       ", out_em[2, 1, end])
println("Milstein final I: ", out_mil[2, 1, end])
```

    EM final I:       1.2322250718903858
    Milstein final I: 1.232328744906324

The Milstein scheme achieves better strong convergence (individual path
accuracy) at the same step size, which matters when comparing specific
realisations.

## Effect of noise magnitude

The parameter $\sigma$ controls the noise intensity:

``` julia
for σ_val in [0.01, 0.1, 0.5]
    p = (beta=0.5, gamma=0.1, sigma=σ_val, I0=10.0, N=1000.0)
    s = dust_system_create(sir_sde, p; n_particles=50, dt=0.01, seed=42)
    dust_system_set_state_initial!(s)
    out = dust_system_simulate(s, times)
    I_var = var(out[2, :, argmax(vec(mean(out[2, :, :], dims=1)))])
    println("σ = $σ_val → variance at peak: $(round(I_var, digits=2))")
end
```

    σ = 0.01 → variance at peak: 0.15
    σ = 0.1 → variance at peak: 14.43
    σ = 0.5 → variance at peak: 352.79

Higher $\sigma$ produces wider spread in trajectories.

## Summary

| Feature | Syntax | Solver |
|----|----|----|
| ODE (deterministic) | `deriv(X) = expr` | DP5, SDIRK |
| Discrete stochastic | `update(X) = expr` | Fixed dt |
| SDE (continuous stochastic) | `deriv(X) = expr` + `diffusion(X) = expr` | Euler-Maruyama, Milstein |

Key points:

- `diffusion(X)` specifies the noise coefficient $\sigma_X$ for state
  variable $X$
- Every `diffusion()` variable must have a matching `deriv()` (drift)
- States without `diffusion()` evolve deterministically
- Use small `dt` for accuracy; SDE solvers use fixed time steps
- Multiple particles give independent realisations (each with its own
  Wiener process)
- The Milstein scheme offers better accuracy than Euler-Maruyama at the
  same step size
