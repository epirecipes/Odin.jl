# Delay Compartments for Erlang-Distributed Durations


## Introduction

Many infectious disease models assume exponentially distributed sojourn
times in each compartment. This is mathematically convenient but
biologically unrealistic: real latent and infectious periods are
concentrated around a mean duration, not memoryless.

**Delay compartments** (also called the *linear chain trick*) address
this by splitting a single compartment into a chain of $k$
sub-compartments. An individual must pass through all $k$ stages
sequentially, producing an **Erlang-distributed** total sojourn time:

| Property     | Formula                         |
|--------------|---------------------------------|
| Distribution | $\text{Erlang}(k,\; k\sigma)$   |
| Mean         | $1/\sigma$ (independent of $k$) |
| Variance     | $1/(k\sigma^2)$                 |
| CV           | $1/\sqrt{k}$                    |

- At $k = 1$ we recover the standard **exponential** distribution
  (maximum variance).
- As $k$ increases the distribution concentrates around the mean,
  approaching a **fixed delay** as $k \to \infty$.

This technique is widely used in malaria modelling (for the extrinsic
incubation period in mosquitoes), influenza (serial interval shaping),
and any process where the shape of the sojourn-time distribution
materially affects transmission dynamics.

## Model: SEIR with Delay Compartments

We define a stochastic discrete-time SEIR model where:

- The **latent (E)** compartment is a chain of `k_E` sub-compartments,
  each with transition rate `k_E * sigma` so the mean latent period
  remains `1/sigma`.
- The **infectious (I)** compartment is a chain of `k_I`
  sub-compartments, each with rate `k_I * gamma`, preserving mean
  infectious period `1/gamma`.

``` julia
using Odin
using Plots
using Statistics

seir_delay = @odin begin
    # Dimensions for delay compartment chains
    dim(E) = k_E
    dim(I) = k_I
    dim(n_EE) = k_E
    dim(n_II) = k_I

    # Force of infection and stochastic transitions
    n_SE = Binomial(S, 1 - exp(-beta * I_total / N * dt))
    n_EE[i] = Binomial(E[i], 1 - exp(-k_E * sigma * dt))
    n_II[i] = Binomial(I[i], 1 - exp(-k_I * gamma * dt))

    I_total = sum(I)

    # State updates — chain progression
    update(S) = S - n_SE
    update(E[1]) = E[1] + n_SE - n_EE[1]
    update(E[2:k_E]) = E[i] + n_EE[i - 1] - n_EE[i]
    update(I[1]) = I[1] + n_EE[k_E] - n_II[1]
    update(I[2:k_I]) = I[i] + n_II[i - 1] - n_II[i]
    update(R) = R + n_II[k_I]

    # Incidence tracking (daily reset)
    initial(incidence, zero_every = 1) = 0
    update(incidence) = incidence + n_SE

    # Data comparison
    cases = data()
    cases ~ Poisson(incidence + 1e-6)

    # Initial conditions
    initial(S) = N - I0
    initial(E[i]) = 0
    initial(I[1]) = I0
    initial(I[2:k_I]) = 0
    initial(R) = 0

    # Parameters
    beta = parameter(0.8)
    sigma = parameter(0.2)     # 1/sigma = 5-day mean latent period
    gamma = parameter(0.1)     # 1/gamma = 10-day mean infectious period
    k_E = parameter(4)         # latent sub-compartments
    k_I = parameter(4)         # infectious sub-compartments
    I0 = parameter(5)
    N = parameter(10000)
end
```

    Odin.DustSystemGenerator{var"##OdinModel#277"}(var"##OdinModel#277"(0, [:incidence, :S, :E, :I, :R], [:beta, :sigma, :gamma, :k_E, :k_I, :I0, :N], false, false, true, false, false, Dict{Symbol, Array}()))

The key insight is that every sub-compartment in the E chain has rate
$k_E \sigma$, so the total time spent traversing all $k_E$ stages is
$\text{Erlang}(k_E, k_E \sigma)$ with mean $1/\sigma$ regardless of
$k_E$. The same logic applies to the I chain.

## Effect of Chain Length on Epidemic Dynamics

Increasing $k$ sharpens the sojourn-time distribution while keeping the
mean constant. This changes the timing and height of the epidemic peak
because individuals progress through compartments more synchronously.

We compare $k = 1$ (standard exponential), $k = 4$, and $k = 10$:

``` julia
times = collect(0.0:1.0:150.0)
k_values = [1, 4, 10]
colors = [:blue, :red, :green]

p1 = plot(xlabel="Day", ylabel="Mean Total Infected",
          title="Effect of Delay Compartments on Epidemic Curve",
          legend=:topright)

for (idx, k) in enumerate(k_values)
    pars = (beta=0.8, sigma=0.2, gamma=0.1,
            k_E=Float64(k), k_I=Float64(k), I0=5.0, N=10000.0)

    I_mean = zeros(length(times))
    n_runs = 50
    i_start = 2 + k       # first I index: 1 (S) + k (E compartments) + 1
    i_end = 1 + k + k     # last I index:  1 + k_E + k_I

    for seed in 1:n_runs
        sys = System(seir_delay, pars; dt=1.0, seed=seed)
        reset!(sys)
        r = simulate(sys, times)
        for t_idx in 1:length(times)
            I_mean[t_idx] += sum(r[i_start:i_end, 1, t_idx])
        end
    end
    I_mean ./= n_runs
    plot!(p1, times, I_mean, label="k = $k", color=colors[idx], lw=2)
end
p1
```

![](11_delay_model_files/figure-commonmark/cell-3-output-1.svg)

Higher $k$ produces a sharper, slightly earlier peak. The reduced
variance in sojourn times leads to more synchronised progression through
compartments, concentrating the infected population into a narrower
window.

## Stochastic Variability

Individual realisations show how $k$ also affects run-to-run
variability:

``` julia
ps = []
for (idx, k) in enumerate(k_values)
    pars = (beta=0.8, sigma=0.2, gamma=0.1,
            k_E=Float64(k), k_I=Float64(k), I0=5.0, N=10000.0)
    i_start = 2 + k
    i_end = 1 + k + k

    pk = plot(xlabel="Day", ylabel="Total Infected",
              title="k = $k", legend=false, ylim=(0, 3000))

    for seed in 1:20
        sys = System(seir_delay, pars; dt=1.0, seed=seed)
        reset!(sys)
        r = simulate(sys, times)
        I_total = [sum(r[i_start:i_end, 1, t]) for t in 1:length(times)]
        plot!(pk, times, I_total, color=colors[idx], alpha=0.3, lw=0.8)
    end
    push!(ps, pk)
end
plot(ps..., layout=(1, 3), size=(900, 300))
```

![](11_delay_model_files/figure-commonmark/cell-4-output-1.svg)

With $k = 1$ the exponential sojourn times introduce substantial
inter-realisation spread. As $k$ increases, individual trajectories
cluster more tightly around the mean.

## Fitting to Synthetic Data

We now demonstrate parameter recovery: generate data from the model with
known parameters, then estimate $\beta$, $\sigma$, and $\gamma$ using a
particle filter likelihood and MCMC.

### Generate synthetic observations

``` julia
true_pars = (beta=0.8, sigma=0.2, gamma=0.1,
             k_E=4.0, k_I=4.0, I0=5.0, N=10000.0)

obs_times = collect(0.0:1.0:100.0)
sys_true = System(seir_delay, true_pars; dt=1.0, seed=1)
reset!(sys_true)
true_result = simulate(sys_true, obs_times)

# Incidence is the last state: index = 1 (S) + k_E + k_I + 1 (R) + 1 = 11
inc_idx = 3 + 4 + 4
obs_cases = max.(1, round.(Int, true_result[inc_idx, 1, 2:end]))

plot(1:length(obs_cases), obs_cases, seriestype=:bar,
     xlabel="Day", ylabel="Cases", title="Synthetic Observed Data",
     label="", color=:steelblue, alpha=0.7)
```

![](11_delay_model_files/figure-commonmark/cell-5-output-1.svg)

### Set up inference

We fix the chain lengths ($k_E = k_I = 4$) and initial conditions,
estimating only the transmission and progression rates:

``` julia
data = ObservedData(
    [(time=Float64(t), cases=Float64(obs_cases[t]))
     for t in 1:length(obs_cases)]
)

filter = Likelihood(seir_delay, data;
    n_particles=100, dt=1.0, seed=42)

packer = Packer([:beta, :sigma, :gamma];
    fixed=(k_E=4.0, k_I=4.0, I0=5.0, N=10000.0))

likelihood = as_model(filter, packer)

prior = @prior begin
    beta ~ Gamma(4.0, 0.2)
    sigma ~ Gamma(2.0, 0.1)
    gamma ~ Gamma(2.0, 0.05)
end

posterior = likelihood + prior
```

    MontyModel{Odin.var"#monty_model_combine##0#monty_model_combine##1"{MontyModel{Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}, Nothing, Nothing, Nothing}, MontyModel{var"#10#11", var"#12#13"{var"#10#11"}, var"#14#15", Matrix{Float64}}}, Odin.var"#monty_model_combine##4#monty_model_combine##5"{Odin.var"#monty_model_combine##0#monty_model_combine##1"{MontyModel{Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}, Nothing, Nothing, Nothing}, MontyModel{var"#10#11", var"#12#13"{var"#10#11"}, var"#14#15", Matrix{Float64}}}}, Nothing, Matrix{Float64}}(["beta", "sigma", "gamma"], Odin.var"#monty_model_combine##0#monty_model_combine##1"{MontyModel{Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}, Nothing, Nothing, Nothing}, MontyModel{var"#10#11", var"#12#13"{var"#10#11"}, var"#14#15", Matrix{Float64}}}(MontyModel{Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}, Nothing, Nothing, Nothing}(["beta", "sigma", "gamma"], Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}(DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}(Odin.DustSystemGenerator{var"##OdinModel#277"}(var"##OdinModel#277"(0, [:incidence, :S, :E, :I, :R], [:beta, :sigma, :gamma, :k_E, :k_I, :I0, :N], false, false, true, false, false, Dict{Symbol, Array}(:n_EE => [0.0, 0.0, 0.0, 0.0], :n_II => [0.0, 0.0, 0.0, 0.0]))), Odin.FilterData{@NamedTuple{cases::Float64}}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0], [(cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,)  …  (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,)]), 0.0, 100, 1.0, 42, false, nothing), MontyPacker([:beta, :sigma, :gamma], [:beta, :sigma, :gamma], Symbol[], Dict{Symbol, Tuple}(), Dict{Symbol, UnitRange{Int64}}(:beta => 1:1, :gamma => 3:3, :sigma => 2:2), 3, (k_E = 4.0, k_I = 4.0, I0 = 5.0, N = 10000.0), nothing)), nothing, nothing, nothing, Odin.MontyModelProperties(false, false, true, false)), MontyModel{var"#10#11", var"#12#13"{var"#10#11"}, var"#14#15", Matrix{Float64}}(["beta", "sigma", "gamma"], var"#10#11"(), var"#12#13"{var"#10#11"}(var"#10#11"()), var"#14#15"(), [0.0 Inf; 0.0 Inf; 0.0 Inf], Odin.MontyModelProperties(true, true, false, false))), Odin.var"#monty_model_combine##4#monty_model_combine##5"{Odin.var"#monty_model_combine##0#monty_model_combine##1"{MontyModel{Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}, Nothing, Nothing, Nothing}, MontyModel{var"#10#11", var"#12#13"{var"#10#11"}, var"#14#15", Matrix{Float64}}}}(Odin.var"#monty_model_combine##0#monty_model_combine##1"{MontyModel{Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}, Nothing, Nothing, Nothing}, MontyModel{var"#10#11", var"#12#13"{var"#10#11"}, var"#14#15", Matrix{Float64}}}(MontyModel{Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}, Nothing, Nothing, Nothing}(["beta", "sigma", "gamma"], Odin.var"#dust_likelihood_monty##0#dust_likelihood_monty##1"{DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}, MontyPacker}(DustFilter{var"##OdinModel#277", Float64, @NamedTuple{cases::Float64}}(Odin.DustSystemGenerator{var"##OdinModel#277"}(var"##OdinModel#277"(0, [:incidence, :S, :E, :I, :R], [:beta, :sigma, :gamma, :k_E, :k_I, :I0, :N], false, false, true, false, false, Dict{Symbol, Array}(:n_EE => [0.0, 0.0, 0.0, 0.0], :n_II => [0.0, 0.0, 0.0, 0.0]))), Odin.FilterData{@NamedTuple{cases::Float64}}([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0], [(cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,), (cases = 1.0,)  …  (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,), (cases = 9999.0,)]), 0.0, 100, 1.0, 42, false, nothing), MontyPacker([:beta, :sigma, :gamma], [:beta, :sigma, :gamma], Symbol[], Dict{Symbol, Tuple}(), Dict{Symbol, UnitRange{Int64}}(:beta => 1:1, :gamma => 3:3, :sigma => 2:2), 3, (k_E = 4.0, k_I = 4.0, I0 = 5.0, N = 10000.0), nothing)), nothing, nothing, nothing, Odin.MontyModelProperties(false, false, true, false)), MontyModel{var"#10#11", var"#12#13"{var"#10#11"}, var"#14#15", Matrix{Float64}}(["beta", "sigma", "gamma"], var"#10#11"(), var"#12#13"{var"#10#11"}(var"#10#11"()), var"#14#15"(), [0.0 Inf; 0.0 Inf; 0.0 Inf], Odin.MontyModelProperties(true, true, false, false)))), nothing, [0.0 Inf; 0.0 Inf; 0.0 Inf], Odin.MontyModelProperties(true, false, true, false))

### Run MCMC

``` julia
vcv = [0.005 0.0  0.0;
       0.0   0.002 0.0;
       0.0   0.0  0.001]
sampler = random_walk(vcv)

initial_theta = reshape([0.6, 0.15, 0.08], 3, 1)
samples = sample(posterior, sampler, 3000;
    initial=initial_theta, n_chains=1)
```

    Odin.MontySamples([0.6 0.648356287486356 … 0.6142633577334465 0.6142633577334465; 0.15 0.15755402497282567 … 0.0279239043866717 0.0279239043866717; 0.08 0.1105313038292357 … 0.013373532856527046 0.013373532856527046;;;], [-7.495934924878691e6; -7.483557688649121e6; … ; -1.3384643683627895e6; -1.3384643683627895e6;;], [0.6; 0.15; 0.08;;], ["beta", "sigma", "gamma"], Dict{Symbol, Any}(:acceptance_rate => [0.006666666666666667]))

### Posterior distributions

``` julia
burnin = 500
beta_post  = samples.pars[1, burnin:end, 1]
sigma_post = samples.pars[2, burnin:end, 1]
gamma_post = samples.pars[3, burnin:end, 1]

println("Parameter estimates (posterior mean ± std):")
for (name, vals, truth) in [("β", beta_post, 0.8),
                             ("σ", sigma_post, 0.2),
                             ("γ", gamma_post, 0.1)]
    println("  $name: $(round(mean(vals), digits=3)) ± " *
            "$(round(std(vals), digits=3)) (true: $truth)")
end
```

    Parameter estimates (posterior mean ± std):
      β: 0.602 ± 0.012 (true: 0.8)
      σ: 0.028 ± 0.001 (true: 0.2)
      γ: 0.015 ± 0.004 (true: 0.1)

``` julia
p1 = histogram(beta_post, bins=30, normalize=true, label="",
               xlabel="β", ylabel="Density", title="Posterior: β")
vline!(p1, [0.8], color=:red, lw=2, label="True")

p2 = histogram(sigma_post, bins=30, normalize=true, label="",
               xlabel="σ", ylabel="Density", title="Posterior: σ")
vline!(p2, [0.2], color=:red, lw=2, label="True")

p3 = histogram(gamma_post, bins=30, normalize=true, label="",
               xlabel="γ", ylabel="Density", title="Posterior: γ")
vline!(p3, [0.1], color=:red, lw=2, label="True")

plot(p1, p2, p3, layout=(1, 3), size=(900, 300))
```

![](11_delay_model_files/figure-commonmark/cell-9-output-1.svg)

### Trace plots

``` julia
p1 = plot(beta_post, xlabel="Iteration", ylabel="β",
          title="Trace: β", label="")
hline!(p1, [0.8], color=:red, lw=1.5, label="True")

p2 = plot(sigma_post, xlabel="Iteration", ylabel="σ",
          title="Trace: σ", label="")
hline!(p2, [0.2], color=:red, lw=1.5, label="True")

p3 = plot(gamma_post, xlabel="Iteration", ylabel="γ",
          title="Trace: γ", label="")
hline!(p3, [0.1], color=:red, lw=1.5, label="True")

plot(p1, p2, p3, layout=(3, 1), size=(700, 500))
```

![](11_delay_model_files/figure-commonmark/cell-10-output-1.svg)

## Summary

| Feature | Description |
|----|----|
| Delay compartments | Chain of $k$ sub-compartments for Erlang-distributed sojourn times |
| `dim(E) = k_E` | Parameterised array dimensions |
| Erlang($k$, $k\sigma$) | Mean $1/\sigma$ preserved; variance $1/(k\sigma^2)$ decreases with $k$ |
| Particle filter | Likelihood estimation for stochastic model |
| MCMC | Posterior inference on $\beta$, $\sigma$, $\gamma$ |

The delay compartment technique is essential for realistic disease
modelling — particularly for malaria (extrinsic incubation period),
influenza (serial interval), and any process where the exponential
assumption is too dispersed.
