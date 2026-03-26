# Categorical Models

The Categorical module enables compositional model building using ideas from applied category theory. Define simple sub-models, compose them via shared interfaces, and stratify across populations — all without writing monolithic model code.

## Overview

This module provides:

- **`EpiNet`** — a labelled epidemiological network (species + transitions)
- **`compose()`** — combine sub-models via shared species
- **`stratify()`** — replicate a model across groups (age, region, etc.)
- **`lower()`** — convert an `EpiNet` to an `@odin`-compatible expression

The design follows the pattern of [AlgebraicPetri.jl](https://github.com/AlgebraicJulia/AlgebraicPetri.jl) and [CategoricalProjectionModels.jl](https://github.com/sdwfrost/CategoricalProjectionModels.jl).

## Built-in Models

Several standard epidemiological models are provided:

```julia
net = SIR()       # S → I → R
net = SEIR()      # S → E → I → R
net = SIS()       # S → I → S
net = SIRS()      # S → I → R → S
net = SEIRS()     # S → E → I → R → S
net = SIRVax()   # SIR with vaccinated compartments
net = two_strain_SIR()  # Two-strain SIR
```

## Building Custom Networks

```julia
net = EpiNet()
add_species!(net, :S, 990)
add_species!(net, :I, 10)
add_species!(net, :R, 0)
add_transition!(net, :infection, [:S, :I], [:I, :I], :beta)
add_transition!(net, :recovery, [:I], [:R], :gamma)
```

### Querying

```julia
species_names(net)          # [:S, :I, :R]
nspecies(net)                # 3
transition_names(net)        # [:infection, :recovery]
ntransitions(net)            # 2
stoichiometry_matrix(net)    # net change per transition
input_matrix(net)            # input stoichiometry
```

## Composition

Compose two sub-models by identifying shared species:

```julia
# Infection sub-model
infection = EpiNet()
add_species!(infection, :S, 990)
add_species!(infection, :I, 10)
add_transition!(infection, :infect, [:S, :I], [:I, :I], :beta)

# Recovery sub-model
recovery = EpiNet()
add_species!(recovery, :I, 10)
add_species!(recovery, :R, 0)
add_transition!(recovery, :recover, [:I], [:R], :gamma)

# Compose via shared :I
full = compose(infection, recovery, [:I])
```

### Interface-based Composition

```julia
full = compose_with_interface(
    [infection, recovery],
    Dict(:I => [1, 2]),  # species :I appears in both sub-models
)
```

## Stratification

Replicate a model across multiple groups with optional inter-group coupling:

```julia
# Base SIR model
base = SIR()

# 3-age-group stratification
strat = stratify(base, 3;
    mixing = [0.8 0.15 0.05;
              0.15 0.7 0.15;
              0.05 0.15 0.8],
)
```

This creates an age-structured SIR with 9 compartments (3 ages × 3 states) and appropriate coupling.

## Lowering to @odin

Convert an `EpiNet` to executable model code:

```julia
# Get @odin expression
expr = lower_expr(net; model_type=:ode)

# Or directly create a generator
gen = lower(net; model_type=:ode)

# Simulate
sys = System(gen, (beta=0.5, gamma=0.1))
reset!(sys)
result = simulate(sys, 0:0.1:100)
```

### Model Types

- `:ode` — continuous ODE model (uses `deriv`)
- `:discrete` — discrete-time stochastic model (uses `update` with `Binomial`)

## Example: Composing an SEIR from Sub-models

```julia
# Exposure sub-model
exposure = EpiNet()
add_species!(exposure, :S, 990)
add_species!(exposure, :E, 0)
add_species!(exposure, :I, 10)
add_transition!(exposure, :expose, [:S, :I], [:E, :I], :beta)

# Progression sub-model
progression = EpiNet()
add_species!(progression, :E, 0)
add_species!(progression, :I, 10)
add_transition!(progression, :progress, [:E], [:I], :sigma)

# Recovery sub-model
recovery = EpiNet()
add_species!(recovery, :I, 10)
add_species!(recovery, :R, 0)
add_transition!(recovery, :recover, [:I], [:R], :gamma)

# Compose
seir = compose(exposure, compose(progression, recovery, [:I]), [:E, :I])
gen = lower(seir; model_type=:ode)
```

## API Reference

```@docs
Odin.EpiNet
Odin.add_species!
Odin.add_transition!
Odin.species_names
Odin.species_concentrations
Odin.nspecies
Odin.transition_names
Odin.transition_rates
Odin.ntransitions
Odin.input_species
Odin.output_species
Odin.stoichiometry_matrix
Odin.input_matrix
Odin.compose
Odin.compose_with_interface
Odin.stratify
Odin.lower
Odin.lower_expr
Odin.sir_net
Odin.seir_net
Odin.sis_net
Odin.sirs_net
Odin.seirs_net
Odin.sir_vax_net
Odin.two_strain_sir_net
```
