# Bayesian Inference with MCMC


## Introduction

Full Bayesian inference with particle filter + MCMC using
odin2/dust2/monty.

## Model and Data

``` r
library(odin2)
library(dust2)
library(monty)

sir <- odin({
  update(S) <- S - n_SI
  update(I) <- I + n_SI - n_IR
  update(R) <- R + n_IR
  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0
  initial(incidence, zero_every = 1) <- 0
  update(incidence) <- incidence + n_SI

  p_SI <- 1 - exp(-beta * I / N * dt)
  p_IR <- 1 - exp(-gamma * dt)
  n_SI <- Binomial(S, p_SI)
  n_IR <- Binomial(I, p_IR)

  cases <- data()
  cases ~ Poisson(incidence + 1e-6)

  beta <- parameter(0.5)
  gamma <- parameter(0.1)
  I0 <- parameter(10)
  N <- parameter(1000)
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

    ℹ Re-compiling odin.system8cef4206

    ── R CMD INSTALL ───────────────────────────────────────────────────────────────
    * installing *source* package ‘odin.system8cef4206’ ...
    ** this is package ‘odin.system8cef4206’ version ‘0.0.1’
    ** using staged installation
    ** libs
    using C++ compiler: ‘Homebrew clang version 21.1.5’
    using SDK: ‘MacOSX15.5.sdk’
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c cpp11.cpp -o cpp11.o
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c dust.cpp -o dust.o
    In file included from dust.cpp:99:
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
    dust.cpp:105:20: note: in instantiation of function template specialization 'dust2::r::dust2_discrete_alloc<odin_system>' requested here
      105 |   return dust2::r::dust2_discrete_alloc<odin_system>(r_pars, r_time, r_time_control, r_n_particles, r_n_groups, r_seed, r_deterministic, r_n_threads);
          |                    ^
    2 warnings generated.
    clang++ -arch arm64 -std=gnu++17 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -L/Library/Frameworks/R.framework/Resources/lib -L/opt/R/arm64/lib -o odin.system8cef4206.so cpp11.o dust.o -F/Library/Frameworks/R.framework/.. -framework R
    installing to /private/var/folders/yh/30rj513j6mn1n7x556c2v4w80000gn/T/RtmpGNwryX/devtools_install_4ffa234b675e/00LOCK-dust_4ffa71be61f9/00new/odin.system8cef4206/libs
    ** checking absolute paths in shared objects and dynamic libraries
    * DONE (odin.system8cef4206)

    ℹ Loading odin.system8cef4206

## Generate Synthetic Data

``` r
true_pars <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
times <- seq(0, 50, by = 1)

sys <- dust_system_create(sir, true_pars, dt = 1, seed = 1)
dust_system_set_state_initial(sys)
obs_result <- dust_system_simulate(sys, times)
observed <- round(obs_result[4, -1])

data <- data.frame(time = times[-1], cases = observed)
```

## Set Up Inference

``` r
filter <- dust_filter_create(sir, time_start = 0, data = data,
                             n_particles = 200, seed = 42)

packer <- monty_packer(c("beta", "gamma"),
                       fixed = list(I0 = 10, N = 1000))

likelihood <- dust_likelihood_monty(filter, packer)

prior <- monty_dsl({
  beta ~ Gamma(shape = 2, rate = 4)
  gamma ~ Gamma(shape = 2, rate = 20)
})

posterior <- likelihood + prior
```

## Run MCMC

``` r
vcv <- matrix(c(0.005, 0, 0, 0.001), 2, 2)
sampler <- monty_sampler_random_walk(vcv)

samples <- monty_sample(posterior, sampler, 2000, initial = c(0.4, 0.08))
```

    ⡀⠀ Sampling  ■                                |   0% ETA: 39s

    ⠄⠀ Sampling  ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■   |  96% ETA:  0s

    ✔ Sampled 2000 steps across 1 chain in 3.1s

## Results

``` r
beta_samples <- samples$pars[1, 500:2000, 1]
gamma_samples <- samples$pars[2, 500:2000, 1]

cat("β: mean =", round(mean(beta_samples), 3),
    ", 95% CI = [", round(quantile(beta_samples, 0.025), 3),
    ",", round(quantile(beta_samples, 0.975), 3), "]\n")
```

    β: mean = 0.493 , 95% CI = [ 0.436 , 0.568 ]

``` r
cat("γ: mean =", round(mean(gamma_samples), 3),
    ", 95% CI = [", round(quantile(gamma_samples, 0.025), 3),
    ",", round(quantile(gamma_samples, 0.975), 3), "]\n")
```

    γ: mean = 0.116 , 95% CI = [ 0.075 , 0.173 ]

``` r
cat("True: β = 0.5, γ = 0.1\n")
```

    True: β = 0.5, γ = 0.1

``` r
par(mfrow = c(1, 2))
hist(beta_samples, breaks = 30, main = "Posterior: β", xlab = "β", probability = TRUE)
abline(v = 0.5, col = "red", lwd = 2)

hist(gamma_samples, breaks = 30, main = "Posterior: γ", xlab = "γ", probability = TRUE)
abline(v = 0.1, col = "red", lwd = 2)
```

![](06_inference_files/figure-commonmark/unnamed-chunk-6-1.png)

## Trace Plots

``` r
par(mfrow = c(2, 1), mar = c(4, 4, 2, 1))
plot(beta_samples, type = "l", xlab = "Iteration", ylab = "β", main = "Trace: β")
abline(h = 0.5, col = "red")
plot(gamma_samples, type = "l", xlab = "Iteration", ylab = "γ", main = "Trace: γ")
abline(h = 0.1, col = "red")
```

![](06_inference_files/figure-commonmark/unnamed-chunk-7-1.png)
