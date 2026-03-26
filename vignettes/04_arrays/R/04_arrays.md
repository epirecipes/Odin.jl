# Age-Structured SIR with Arrays


## Introduction

Age-structured SIR model using arrays in odin2.

## Model Definition

``` r
library(odin2)
library(dust2)

sir_age <- odin({
  n_age <- parameter(3)
  dim(S) <- n_age
  dim(I) <- n_age
  dim(R) <- n_age
  dim(beta) <- n_age
  dim(S0) <- n_age
  dim(I0) <- n_age

  deriv(S[]) <- -beta[i] * S[i] * total_I / N
  deriv(I[]) <- beta[i] * S[i] * total_I / N - gamma * I[i]
  deriv(R[]) <- gamma * I[i]

  total_I <- sum(I)

  initial(S[]) <- S0[i]
  initial(I[]) <- I0[i]
  initial(R[]) <- 0

  S0[] <- parameter()
  I0[] <- parameter()
  beta[] <- parameter()
  gamma <- parameter(0.1)
  N <- parameter(1000)
})
```

    Warning in odin({: Found 3 compatibility issues
    Drop arrays from lhs of assignments from 'parameter()'
    ✖ S0[] <- parameter()
    ✔ S0 <- parameter()
    ✖ I0[] <- parameter()
    ✔ I0 <- parameter()
    ✖ beta[] <- parameter()
    ✔ beta <- parameter()

    ✔ Wrote 'DESCRIPTION'

    ✔ Wrote 'NAMESPACE'

    ✔ Wrote 'R/dust.R'

    ✔ Wrote 'src/dust.cpp'

    ✔ Wrote 'src/Makevars'

    ℹ 13 functions decorated with [[cpp11::register]]

    ✔ generated file 'cpp11.R'

    ✔ generated file 'cpp11.cpp'

    ℹ Re-compiling odin.system9375e197

    ── R CMD INSTALL ───────────────────────────────────────────────────────────────
    * installing *source* package ‘odin.system9375e197’ ...
    ** this is package ‘odin.system9375e197’ version ‘0.0.1’
    ** using staged installation
    ** libs
    using C++ compiler: ‘Homebrew clang version 21.1.5’
    using SDK: ‘MacOSX15.5.sdk’
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c cpp11.cpp -o cpp11.o
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c dust.cpp -o dust.o
    In file included from dust.cpp:111:
    In file included from /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include/dust2/r/continuous/system.hpp:4:
    /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include/monty/r/random.hpp:60:43: warning: implicit conversion from 'type' (aka 'unsigned long') to 'double' changes value from 18446744073709551615 to 18446744073709551616 [-Wimplicit-const-int-float-conversion]
       60 |       std::ceil(std::abs(::unif_rand()) * std::numeric_limits<size_t>::max());
          |                                         ~ ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include/monty/r/random.hpp:60:43: warning: implicit conversion from 'type' (aka 'unsigned long') to 'double' changes value from 18446744073709551615 to 18446744073709551616 [-Wimplicit-const-int-float-conversion]
       60 |       std::ceil(std::abs(::unif_rand()) * std::numeric_limits<size_t>::max());
          |                                         ~ ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include/dust2/r/continuous/system.hpp:34:33: note: in instantiation of function template specialization 'monty::random::r::as_rng_seed<monty::random::xoshiro_state<unsigned long long, 4, monty::random::scrambler::plus>>' requested here
       34 |   auto seed = monty::random::r::as_rng_seed<rng_state_type>(r_seed);
          |                                 ^
    dust.cpp:115:20: note: in instantiation of function template specialization 'dust2::r::dust2_continuous_alloc<odin_system>' requested here
      115 |   return dust2::r::dust2_continuous_alloc<odin_system>(r_pars, r_time, r_time_control, r_n_particles, r_n_groups, r_seed, r_deterministic, r_n_threads);
          |                    ^
    2 warnings generated.
    clang++ -arch arm64 -std=gnu++17 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -L/Library/Frameworks/R.framework/Resources/lib -L/opt/R/arm64/lib -o odin.system9375e197.so cpp11.o dust.o -F/Library/Frameworks/R.framework/.. -framework R
    installing to /private/var/folders/yh/30rj513j6mn1n7x556c2v4w80000gn/T/RtmpXCoVbg/devtools_install_4dcf716c095f/00LOCK-dust_4dcf345479dc/00new/odin.system9375e197/libs
    ** checking absolute paths in shared objects and dynamic libraries
    * DONE (odin.system9375e197)

    ℹ Loading odin.system9375e197

## Parameters and Simulation

``` r
pars <- list(
  n_age = 3,
  S0 = c(300, 500, 190),
  I0 = c(5, 3, 2),
  beta = c(0.4, 0.3, 0.2),
  gamma = 0.1,
  N = 1000
)

sys <- System(sir_age, pars, ode_control = dust_ode_control())
dust_system_set_state_initial(sys)
times <- seq(0, 200, by = 0.5)
result <- simulate(sys, times)
cat("State dimensions:", dim(result), "\n")
```

    State dimensions: 9 401 

## Visualising by Age Group

``` r
age_labels <- c("Children (0-14)", "Adults (15-64)", "Elderly (65+)")
cols <- c("blue", "red", "green")

# Infected by age group
plot(NULL, xlim = range(times), ylim = c(0, max(result[4:6, ])),
     xlab = "Time", ylab = "Infected", main = "Infected by Age Group")
for (i in 1:3) {
  lines(times, result[3 + i, ], col = cols[i], lwd = 2)
}
legend("topright", legend = age_labels, col = cols, lwd = 2)
```

![](04_arrays_files/figure-commonmark/unnamed-chunk-3-1.png)

``` r
par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))
for (g in 1:3) {
  plot(times, result[g, ], type = "l", col = "blue", lwd = 2,
       xlab = "Time", ylab = "Population", main = age_labels[g],
       ylim = c(0, max(result[g, ])))
  lines(times, result[3 + g, ], col = "red", lwd = 2)
  lines(times, result[6 + g, ], col = "green", lwd = 2)
}
```

![](04_arrays_files/figure-commonmark/unnamed-chunk-4-1.png)

## Final Attack Rates

``` r
for (g in 1:3) {
  R_final <- result[6 + g, length(times)]
  N_group <- pars$S0[g] + pars$I0[g]
  ar <- 100 * R_final / N_group
  cat(age_labels[g], ": Attack rate =", round(ar, 1), "%\n")
}
```

    Children (0-14) : Attack rate = 97.6 %
    Adults (15-64) : Attack rate = 93.9 %
    Elderly (65+) : Attack rate = 84.7 %
