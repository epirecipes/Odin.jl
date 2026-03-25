# Particle Filtering in R (CPU)


## Overview

The Julia vignette (vignette 31) demonstrates GPU-accelerated particle
filtering via Odin.jl’s Metal/CUDA backend system. **R does not have an
equivalent GPU particle filter** in the odin2/dust2 ecosystem, but the
CPU-based `dust2` particle filter is highly optimised with C++ and
OpenMP parallelisation.

This vignette shows the equivalent R workflow for particle filtering
using `odin2` and `dust2`, which serves as the CPU baseline.

## SIR Particle Filter in R

### Define the model

``` r
library(odin2)
library(dust2)

sir <- odin2::odin({
  update(S) <- S - n_SI
  update(I) <- I + n_SI - n_IR
  update(R) <- R + n_IR
  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0

  p_SI <- 1 - exp(-beta * I / N * dt)
  p_IR <- 1 - exp(-gamma * dt)
  n_SI <- Binomial(S, p_SI)
  n_IR <- Binomial(I, p_IR)

  cases <- data()
  cases ~ Poisson(max(I, 1))

  N <- parameter(1000)
  I0 <- parameter(10)
  beta <- parameter(0.5)
  gamma <- parameter(0.1)
})
```

    ✔ Wrote 'DESCRIPTION'

    ✔ Wrote 'NAMESPACE'

    ✔ Wrote 'R/dust.R'

    ✔ Wrote 'src/dust.cpp'

    ✔ Wrote 'src/Makevars'

    ℹ 27 functions decorated with [[cpp11::register]]

    ✔ generated file 'cpp11.R'

    ✔ generated file 'cpp11.cpp'

    ℹ Re-compiling odin.system7004d4fd

    ── R CMD INSTALL ───────────────────────────────────────────────────────────────
    * installing *source* package ‘odin.system7004d4fd’ ...
    ** this is package ‘odin.system7004d4fd’ version ‘0.0.1’
    ** using staged installation
    ** libs
    using C++ compiler: ‘Homebrew clang version 21.1.5’
    using SDK: ‘MacOSX15.5.sdk’
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c cpp11.cpp -o cpp11.o
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c dust.cpp -o dust.o
    In file included from dust.cpp:92:
    In file included from /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include/dust2/r/discrete/system.hpp:5:
    /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include/monty/r/random.hpp:60:43: warning: implicit conversion from 'type' (aka 'unsigned long') to 'double' changes value from 18446744073709551615 to 18446744073709551616 [-Wimplicit-const-int-float-conversion]
       60 |       std::ceil(std::abs(::unif_rand()) * std::numeric_limits<size_t>::max());
          |                                         ~ ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include/monty/r/random.hpp:60:43: warning: implicit conversion from 'type' (aka 'unsigned long') to 'double' changes value from 18446744073709551615 to 18446744073709551616 [-Wimplicit-const-int-float-conversion]
       60 |       std::ceil(std::abs(::unif_rand()) * std::numeric_limits<size_t>::max());
          |                                         ~ ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include/dust2/r/discrete/system.hpp:41:33: note: in instantiation of function template specialization 'monty::random::r::as_rng_seed<monty::random::xoshiro_state<unsigned long long, 4, monty::random::scrambler::plus>>' requested here
       41 |   auto seed = monty::random::r::as_rng_seed<rng_state_type>(r_seed);
          |                                 ^
    dust.cpp:98:20: note: in instantiation of function template specialization 'dust2::r::dust2_discrete_alloc<odin_system>' requested here
       98 |   return dust2::r::dust2_discrete_alloc<odin_system>(r_pars, r_time, r_time_control, r_n_particles, r_n_groups, r_seed, r_deterministic, r_n_threads);
          |                    ^
    2 warnings generated.
    clang++ -arch arm64 -std=gnu++17 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -L/Library/Frameworks/R.framework/Resources/lib -L/opt/R/arm64/lib -o odin.system7004d4fd.so cpp11.o dust.o -F/Library/Frameworks/R.framework/.. -framework R
    installing to /private/var/folders/yh/30rj513j6mn1n7x556c2v4w80000gn/T/RtmpIYbjqZ/devtools_install_13b33191bf17d/00LOCK-dust_13b3378261b28/00new/odin.system7004d4fd/libs
    ** checking absolute paths in shared objects and dynamic libraries
    * DONE (odin.system7004d4fd)

    ℹ Loading odin.system7004d4fd

### Generate synthetic data

``` r
pars <- list(N = 1000, I0 = 10, beta = 0.5, gamma = 0.1)

sys <- dust_system_create(sir, pars, n_particles = 1, dt = 1, seed = 1L)
dust_system_set_state_initial(sys)
times <- seq(1, 100, by = 1)
result <- dust_system_simulate(sys, times)

# dust2 returns (n_state, n_particles, n_times) or (n_state, n_times)
# depending on n_particles; extract I (state 2) appropriately
if (length(dim(result)) == 3) {
  cases_raw <- result[2, 1, ]
} else {
  cases_raw <- result[2, ]
}

data_df <- data.frame(
  time = times,
  cases = pmax(1, round(cases_raw))
)
```

### Run the particle filter

``` r
filter <- dust_filter_create(sir, data = data_df, time_start = 0,
                              n_particles = 10000, dt = 1, seed = 42L)
ll <- dust_likelihood_run(filter, pars)
cat(sprintf("Log-likelihood: %.2f\n", ll))
```

    Log-likelihood: -265.55

### Benchmark

``` r
n_reps <- 20
timings <- replicate(n_reps, {
  t0 <- proc.time()["elapsed"]
  dust_likelihood_run(filter, pars)
  proc.time()["elapsed"] - t0
})
cat(sprintf("Particle filter (10,000 particles, %d reps):\n", n_reps))
```

    Particle filter (10,000 particles, 20 reps):

``` r
cat(sprintf("  Mean: %.1f ms\n", mean(timings) * 1000))
```

      Mean: 97.4 ms

``` r
cat(sprintf("  SD:   %.1f ms\n", sd(timings) * 1000))
```

      SD:   1.2 ms

### Integration with monty MCMC

``` r
library(monty)

packer <- monty_packer(c("beta", "gamma"), fixed = list(I0 = 10, N = 1000))
ll_model <- dust_likelihood_monty(filter, packer)

prior <- monty_dsl({
  beta ~ Exponential(mean = 0.5)
  gamma ~ Exponential(mean = 0.2)
})

posterior <- ll_model + prior

sampler <- monty_sampler_random_walk(vcv = diag(c(0.01, 0.001)))
```

Running the full MCMC is commented out for vignette build speed:

``` r
samples <- monty_sample(posterior, sampler, n_steps = 1000, n_chains = 4)
```

## GPU acceleration: R vs Julia

| Feature         | R (dust2)                 | Julia (Odin.jl)               |
|-----------------|---------------------------|-------------------------------|
| Particle filter | `dust_filter_create()`    | `gpu_dust_filter_create()`    |
| Backend         | C++ / OpenMP (CPU)        | Metal / CUDA / CPU            |
| Parallelism     | OpenMP threads            | GPU kernels + CPU resampling  |
| MCMC bridge     | `dust_likelihood_monty()` | `gpu_dust_likelihood_monty()` |

### Why no GPU in R?

R’s C++ compilation model and memory management make GPU kernel
integration challenging. The dust2 C++ code uses OpenMP for CPU-level
parallelism across particles, which provides significant speedup on
multi-core machines without requiring GPU hardware.

For GPU-accelerated particle filtering, use the Julia implementation via
`Odin.jl`, which provides Metal (Apple Silicon) and CUDA (NVIDIA)
backends.

## Summary

- R’s `dust2` particle filter is well-optimised with C++ and OpenMP
- For most workloads, the CPU filter in R is sufficient
- GPU acceleration is available through the Julia implementation
  (`Odin.jl`)
- The R and Julia APIs follow the same conceptual design: create → run →
  bridge to MCMC
