# Basic ODE Model: SIR


## Introduction

This vignette demonstrates how to define and simulate a basic SIR
ordinary differential equation model using the R `odin2`/`dust2`
packages. This is the R companion to the Julia vignette.

## Model Definition

``` r
library(odin2)
library(dust2)

sir <- odin({
  deriv(S) <- -beta * S * I / N
  deriv(I) <- beta * S * I / N - gamma * I
  deriv(R) <- gamma * I
  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0
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

    ℹ 13 functions decorated with [[cpp11::register]]

    ✔ generated file 'cpp11.R'

    ✔ generated file 'cpp11.cpp'

    ℹ Re-compiling odin.system642b6fa1

    ── R CMD INSTALL ───────────────────────────────────────────────────────────────
    * installing *source* package ‘odin.system642b6fa1’ ...
    ** this is package ‘odin.system642b6fa1’ version ‘0.0.1’
    ** using staged installation
    ** libs
    using C++ compiler: ‘Homebrew clang version 21.1.5’
    using SDK: ‘MacOSX15.5.sdk’
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c cpp11.cpp -o cpp11.o
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c dust.cpp -o dust.o
    In file included from dust.cpp:73:
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
    dust.cpp:77:20: note: in instantiation of function template specialization 'dust2::r::dust2_continuous_alloc<odin_system>' requested here
       77 |   return dust2::r::dust2_continuous_alloc<odin_system>(r_pars, r_time, r_time_control, r_n_particles, r_n_groups, r_seed, r_deterministic, r_n_threads);
          |                    ^
    2 warnings generated.
    clang++ -arch arm64 -std=gnu++17 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -L/Library/Frameworks/R.framework/Resources/lib -L/opt/R/arm64/lib -o odin.system642b6fa1.so cpp11.o dust.o -F/Library/Frameworks/R.framework/.. -framework R
    installing to /private/var/folders/yh/30rj513j6mn1n7x556c2v4w80000gn/T/RtmpJDrpvW/devtools_install_49536fb8c2d0/00LOCK-dust_495361b50689/00new/odin.system642b6fa1/libs
    ** checking absolute paths in shared objects and dynamic libraries
    * DONE (odin.system642b6fa1)

    ℹ Loading odin.system642b6fa1

## Simulation

``` r
pars <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
sys <- dust_system_create(sir, pars, ode_control = dust_ode_control())
dust_system_set_state_initial(sys)
times <- seq(0, 200, by = 1)
result <- dust_system_simulate(sys, times)
```

## Plot

``` r
S <- result[1, ]
I <- result[2, ]
R <- result[3, ]

plot(times, S, type = "l", col = "blue", lwd = 2,
     xlab = "Time", ylab = "Population",
     main = "SIR ODE Model", ylim = c(0, 1000))
lines(times, I, col = "red", lwd = 2)
lines(times, R, col = "green", lwd = 2)
legend("right", legend = c("S", "I", "R"),
       col = c("blue", "red", "green"), lwd = 2)
```

![](01_basic_ode_files/figure-commonmark/unnamed-chunk-3-1.png)

## Parameter Exploration

``` r
betas <- c(0.3, 0.5, 0.8)
cols <- c("blue", "red", "darkgreen")

plot(NULL, xlim = c(0, 200), ylim = c(0, 500),
     xlab = "Time", ylab = "Infected",
     main = "Effect of β on Epidemic Dynamics")

for (i in seq_along(betas)) {
  pars_b <- list(beta = betas[i], gamma = 0.1, I0 = 10, N = 1000)
  sys_b <- dust_system_create(sir, pars_b, ode_control = dust_ode_control())
  dust_system_set_state_initial(sys_b)
  res <- dust_system_simulate(sys_b, times)
  lines(times, res[2, ], col = cols[i], lwd = 2)
}
legend("topright", legend = paste("β =", betas), col = cols, lwd = 2)
```

![](01_basic_ode_files/figure-commonmark/unnamed-chunk-4-1.png)

## Verification

``` r
total <- S + I + R
cat("Population at t=0:", total[1], "\n")
```

    Population at t=0: 1000 

``` r
cat("Population at t=200:", tail(total, 1), "\n")
```

    Population at t=200: 1000 

``` r
cat("Max deviation:", max(abs(total - 1000)), "\n")
```

    Max deviation: 5.684342e-13 

## Final Size

``` r
final_R <- tail(R, 1)
cat("Final epidemic size:", round(final_R, 1), "out of", pars$N, "\n")
```

    Final epidemic size: 993.1 out of 1000 

``` r
cat("Attack rate:", round(100 * final_R / pars$N, 1), "%\n")
```

    Attack rate: 99.3 %
