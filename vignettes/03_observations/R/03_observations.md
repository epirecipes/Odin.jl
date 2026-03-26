# Incidence Tracking with zero_every


## Introduction

Tracking incidence (new cases per time period) using the `zero_every`
feature in odin2.

## Model with Incidence Tracking

``` r
library(odin2)
library(dust2)

sir_inc <- odin({
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

    ℹ 12 functions decorated with [[cpp11::register]]

    ✔ generated file 'cpp11.R'

    ✔ generated file 'cpp11.cpp'

    ℹ Re-compiling odin.system0ed821d3

    ── R CMD INSTALL ───────────────────────────────────────────────────────────────
    * installing *source* package ‘odin.system0ed821d3’ ...
    ** this is package ‘odin.system0ed821d3’ version ‘0.0.1’
    ** using staged installation
    ** libs
    using C++ compiler: ‘Homebrew clang version 21.1.5’
    using SDK: ‘MacOSX15.5.sdk’
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c cpp11.cpp -o cpp11.o
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c dust.cpp -o dust.o
    In file included from dust.cpp:85:
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
    dust.cpp:89:20: note: in instantiation of function template specialization 'dust2::r::dust2_discrete_alloc<odin_system>' requested here
       89 |   return dust2::r::dust2_discrete_alloc<odin_system>(r_pars, r_time, r_time_control, r_n_particles, r_n_groups, r_seed, r_deterministic, r_n_threads);
          |                    ^
    2 warnings generated.
    clang++ -arch arm64 -std=gnu++17 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -L/Library/Frameworks/R.framework/Resources/lib -L/opt/R/arm64/lib -o odin.system0ed821d3.so cpp11.o dust.o -F/Library/Frameworks/R.framework/.. -framework R
    installing to /private/var/folders/yh/30rj513j6mn1n7x556c2v4w80000gn/T/RtmpJtejq2/devtools_install_784e1c6e46fd/00LOCK-dust_784e38f0a119/00new/odin.system0ed821d3/libs
    ** checking absolute paths in shared objects and dynamic libraries
    * DONE (odin.system0ed821d3)

    ℹ Loading odin.system0ed821d3

## Simulation

``` r
pars <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
times <- seq(0, 100, by = 1)

sys <- System(sir_inc, pars, n_particles = 5, dt = 1, seed = 42)
dust_system_set_state_initial(sys)
result <- simulate(sys, times)
```

## Visualising Incidence

``` r
par(mfrow = c(2, 1), mar = c(4, 4, 2, 1))

# Prevalence
cols <- adjustcolor(rainbow(5), alpha.f = 0.5)
plot(NULL, xlim = range(times), ylim = c(0, max(result[2, , ])),
     xlab = "Time", ylab = "Count", main = "Prevalence (I)")
for (i in 1:5) lines(times, result[2, i, ], col = cols[i])

# Incidence
plot(NULL, xlim = range(times), ylim = c(0, max(result[4, , ])),
     xlab = "Time", ylab = "Count", main = "Incidence (new cases/day)")
for (i in 1:5) lines(times, result[4, i, ], col = cols[i])
```

![](03_observations_files/figure-commonmark/unnamed-chunk-3-1.png)

## Summary

``` r
mean_inc <- colMeans(result[4, , ])
cat("Peak incidence at day:", times[which.max(mean_inc)], "\n")
```

    Peak incidence at day: 13 

``` r
cat("Peak mean incidence:", round(max(mean_inc), 1), "\n")
```

    Peak mean incidence: 91.6 

``` r
cat("Cumulative incidence:", round(sum(mean_inc), 1), "\n")
```

    Cumulative incidence: 985 
