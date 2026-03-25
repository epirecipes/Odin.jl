# Advanced Model: SEIR with Vaccination and Waning


## Introduction

R companion to the Julia advanced model vignette: stochastic SEIR with
time-varying vaccination, waning immunity, and inference.

``` r
library(odin2)
library(dust2)
library(monty)
```

## Model Definition

``` r
seir_vax <- odin({
  n_SE <- Binomial(S, 1 - exp(-beta * I / N * dt))
  n_EI <- Binomial(E, 1 - exp(-sigma * dt))
  n_IR <- Binomial(I, 1 - exp(-gamma * dt))
  n_RS <- Binomial(R, 1 - exp(-omega * dt))
  n_SV <- Binomial(S, 1 - exp(-nu * dt))
  n_VS <- Binomial(V, 1 - exp(-omega_v * dt))

  update(S) <- S - n_SE - n_SV + n_RS + n_VS
  update(E) <- E + n_SE - n_EI
  update(I) <- I + n_EI - n_IR
  update(R) <- R + n_IR - n_RS
  update(V) <- V + n_SV - n_VS

  initial(cases_inc, zero_every = 1) <- 0
  update(cases_inc) <- cases_inc + n_EI

  initial(S) <- N - E0 - I0
  initial(E) <- E0
  initial(I) <- I0
  initial(R) <- 0
  initial(V) <- 0

  cases <- data()
  cases ~ Poisson(cases_inc + 1e-6)

  nu <- interpolate(nu_time, nu_value, "constant")
  nu_time[] <- parameter()
  nu_value[] <- parameter()
  dim(nu_time, nu_value) <- parameter(rank = 1)

  beta <- parameter(0.4)
  sigma <- parameter(0.2)
  gamma <- parameter(0.1)
  omega <- parameter(0.005)
  omega_v <- parameter(0.01)
  E0 <- parameter(5)
  I0 <- parameter(5)
  N <- parameter(10000)
})
```

    Warning in odin({: Found 2 compatibility issues
    Drop arrays from lhs of assignments from 'parameter()'
    ✖ nu_time[] <- parameter()
    ✔ nu_time <- parameter()
    ✖ nu_value[] <- parameter()
    ✔ nu_value <- parameter()

    ✔ Wrote 'DESCRIPTION'

    ✔ Wrote 'NAMESPACE'

    ✔ Wrote 'R/dust.R'

    ✔ Wrote 'src/dust.cpp'

    ✔ Wrote 'src/Makevars'

    ℹ 27 functions decorated with [[cpp11::register]]

    ✔ generated file 'cpp11.R'

    ✔ generated file 'cpp11.cpp'

    ℹ Re-compiling odin.systemb03c8075

    ── R CMD INSTALL ───────────────────────────────────────────────────────────────
    * installing *source* package ‘odin.systemb03c8075’ ...
    ** this is package ‘odin.systemb03c8075’ version ‘0.0.1’
    ** using staged installation
    ** libs
    using C++ compiler: ‘Homebrew clang version 21.1.5’
    using SDK: ‘MacOSX15.5.sdk’
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c cpp11.cpp -o cpp11.o
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c dust.cpp -o dust.o
    In file included from dust.cpp:144:
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
    dust.cpp:150:20: note: in instantiation of function template specialization 'dust2::r::dust2_discrete_alloc<odin_system>' requested here
      150 |   return dust2::r::dust2_discrete_alloc<odin_system>(r_pars, r_time, r_time_control, r_n_particles, r_n_groups, r_seed, r_deterministic, r_n_threads);
          |                    ^
    2 warnings generated.
    clang++ -arch arm64 -std=gnu++17 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -L/Library/Frameworks/R.framework/Resources/lib -L/opt/R/arm64/lib -o odin.systemb03c8075.so cpp11.o dust.o -F/Library/Frameworks/R.framework/.. -framework R
    installing to /private/var/folders/yh/30rj513j6mn1n7x556c2v4w80000gn/T/RtmpAkYQ67/devtools_install_f4a61565e8e3/00LOCK-dust_f4a62615ea63/00new/odin.systemb03c8075/libs
    ** checking absolute paths in shared objects and dynamic libraries
    * DONE (odin.systemb03c8075)

    ℹ Loading odin.systemb03c8075

## Simulation

``` r
pars <- list(
  beta = 0.4, sigma = 0.2, gamma = 0.1,
  omega = 0.005, omega_v = 0.01,
  E0 = 5, I0 = 5, N = 10000,
  nu_time = c(0, 30), nu_value = c(0, 0.005)
)

sys <- dust_system_create(seir_vax, pars, n_particles = 1, dt = 1, seed = 42)
dust_system_set_state_initial(sys)
times <- seq(0, 365, by = 1)
result <- dust_system_simulate(sys, times)

plot(times, result[1, ], type = "l", col = "blue", lwd = 2,
     xlab = "Day", ylab = "Population",
     main = "SEIR-V (single realisation)", ylim = c(0, 10000))
lines(times, result[2, ], col = "orange", lwd = 1.5)
lines(times, result[3, ], col = "red", lwd = 2)
lines(times, result[4, ], col = "green", lwd = 1.5)
lines(times, result[5, ], col = "purple", lwd = 1.5)
abline(v = 30, lty = 2, col = "gray")
legend("right", legend = c("S", "E", "I", "R", "V"),
       col = c("blue", "orange", "red", "green", "purple"), lwd = 2)
```

![](09_advanced_files/figure-commonmark/unnamed-chunk-3-1.png)

## Comparing Vaccination Scenarios

``` r
scenarios <- list(
  list(label = "No vaccination", nu_t = c(0, 365), nu_v = c(0, 0)),
  list(label = "ν = 0.002", nu_t = c(0, 30), nu_v = c(0, 0.002)),
  list(label = "ν = 0.005", nu_t = c(0, 30), nu_v = c(0, 0.005)),
  list(label = "ν = 0.01",  nu_t = c(0, 30), nu_v = c(0, 0.01))
)

cols <- c("black", "blue", "red", "green")
plot(NULL, xlim = c(0, 365), ylim = c(0, 1500),
     xlab = "Day", ylab = "Mean infected",
     main = "Impact of Vaccination Rate")

for (k in seq_along(scenarios)) {
  sc <- scenarios[[k]]
  I_mean <- rep(0, length(times))
  for (seed in 1:20) {
    p <- pars
    p$nu_time <- sc$nu_t
    p$nu_value <- sc$nu_v
    sys <- dust_system_create(seir_vax, p, n_particles = 1, dt = 1, seed = seed)
    dust_system_set_state_initial(sys)
    r <- dust_system_simulate(sys, times)
    I_mean <- I_mean + r[3, ]
  }
  I_mean <- I_mean / 20
  lines(times, I_mean, col = cols[k], lwd = 2)
}
legend("topright", legend = sapply(scenarios, "[[", "label"),
       col = cols, lwd = 2)
```

![](09_advanced_files/figure-commonmark/unnamed-chunk-4-1.png)

## Summary

The R implementation matches the Julia vignette, demonstrating the same
SEIR-V model with time-varying vaccination, multiple realisations, and
scenario comparison.
