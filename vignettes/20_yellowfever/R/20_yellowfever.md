# Yellow Fever SEIRV: Age-Structured Model with Spillover (R)


## Introduction

This is the R companion to the Julia Yellow Fever SEIRV vignette. It
uses odin2/dust2 to define and simulate the same age-structured
stochastic model with spillover force of infection, time-varying R0, and
vaccination.

``` r
library(odin2)
library(dust2)
```

## Model Definition

The model uses odin2 syntax with `update()` equations for discrete-time
stochastic dynamics. Key features include partial array updates
(different rules for age group 1 vs 2:N_age) and `interpolate()` for
time-varying parameters.

``` r
yf_seirv <- odin({
  # === Configuration ===
  N_age <- parameter(5)

  # === Dimensions ===
  dim(S) <- N_age
  dim(E) <- N_age
  dim(I) <- N_age
  dim(R) <- N_age
  dim(V) <- N_age
  dim(C) <- N_age

  dim(S_0) <- N_age
  dim(E_0) <- N_age
  dim(I_0) <- N_age
  dim(R_0) <- N_age
  dim(V_0) <- N_age

  dim(dP1) <- N_age
  dim(dP2) <- N_age
  dim(E_new) <- N_age
  dim(I_new) <- N_age
  dim(R_new) <- N_age
  dim(P_nV) <- N_age
  dim(inv_P_nV) <- N_age
  dim(P) <- N_age
  dim(inv_P) <- N_age
  dim(vacc_eff) <- N_age

  # === Epidemiological rates ===
  t_latent <- parameter(5.0)
  t_infectious <- parameter(5.0)
  rate1 <- 1.0 / t_latent
  rate2 <- 1.0 / t_infectious

  # === Time-varying R0 and spillover ===
  R0_t <- interpolate(R0_time, R0_value, "linear")
  FOI_sp <- interpolate(sp_time, sp_value, "linear")
  beta <- R0_t / t_infectious

  # === Force of infection ===
  P_total <- sum(P)
  I_total <- sum(I)
  FOI_raw <- beta * I_total / max(P_total, 1.0) + FOI_sp
  FOI_max <- 1.0
  FOI_sum <- min(FOI_max, FOI_raw)

  # === Population totals per age ===
  P_nV[1:N_age] <- max(S[i] + R[i], 1e-99)
  inv_P_nV[1:N_age] <- 1.0 / P_nV[i]
  P[1:N_age] <- max(P_nV[i] + V[i], 1e-99)
  inv_P[1:N_age] <- 1.0 / P[i]

  # === Transitions ===
  p_inf <- 1 - exp(-FOI_sum * dt)
  p_lat <- 1 - exp(-rate1 * dt)
  p_rec <- 1 - exp(-rate2 * dt)

  E_new[1:N_age] <- Binomial(S[i], p_inf)
  I_new[1:N_age] <- Binomial(E[i], p_lat)
  R_new[1:N_age] <- Binomial(I[i], p_rec)

  # === Vaccination ===
  vaccine_efficacy <- parameter(0.95)
  dim(vacc_rate) <- N_age
  vacc_eff[1:N_age] <- vacc_rate[i] * vaccine_efficacy * dt

  # === Demographic flows ===
  dP1_rate <- interpolate(dP1_time, dP1_value, "constant")
  dP2_rate <- interpolate(dP2_time, dP2_value, "constant")
  dP1[1:N_age] <- dP1_rate * 0.01
  dP2[1:N_age] <- dP2_rate * 0.01

  # === State updates (grouped by compartment for odin2 contiguity) ===
  update(S[1]) <- max(0.0, S[1] - E_new[1]
                      - vacc_eff[1] * S[1] * inv_P_nV[1]
                      + dP1[1]
                      - dP2[1] * S[1] * inv_P[1])
  update(S[2:N_age]) <- max(0.0, S[i] - E_new[i]
                            - vacc_eff[i] * S[i] * inv_P_nV[i]
                            + dP1[i] * S[i - 1] * inv_P[i - 1]
                            - dP2[i] * S[i] * inv_P[i])

  update(E[1]) <- max(0.0, E[1] + E_new[1] - I_new[1])
  update(E[2:N_age]) <- max(0.0, E[i] + E_new[i] - I_new[i])

  update(I[1]) <- max(0.0, I[1] + I_new[1] - R_new[1])
  update(I[2:N_age]) <- max(0.0, I[i] + I_new[i] - R_new[i])

  update(R[1]) <- max(0.0, R[1] + R_new[1]
                      - vacc_eff[1] * R[1] * inv_P_nV[1]
                      - dP2[1] * R[1] * inv_P[1])
  update(R[2:N_age]) <- max(0.0, R[i] + R_new[i]
                            - vacc_eff[i] * R[i] * inv_P_nV[i]
                            + dP1[i] * R[i - 1] * inv_P[i - 1]
                            - dP2[i] * R[i] * inv_P[i])

  update(V[1]) <- max(0.0, V[1] + vacc_eff[1]
                      - dP2[1] * V[1] * inv_P[1])
  update(V[2:N_age]) <- max(0.0, V[i] + vacc_eff[i]
                            + dP1[i] * V[i - 1] * inv_P[i - 1]
                            - dP2[i] * V[i] * inv_P[i])

  # === Cumulative new cases per step (reset each step) ===
  initial(C[1:N_age], zero_every = 1) <- 0
  update(C[1:N_age]) <- C[i] + I_new[i]

  # === Initial conditions ===
  initial(S[1:N_age]) <- S_0[i]
  initial(E[1:N_age]) <- E_0[i]
  initial(I[1:N_age]) <- I_0[i]
  initial(R[1:N_age]) <- R_0[i]
  initial(V[1:N_age]) <- V_0[i]

  # === Tracked quantities (as state variables, after SEIRVC) ===
  initial(FOI_total) <- 0
  update(FOI_total) <- FOI_sum
  initial(total_I) <- 0
  update(total_I) <- I_total
  initial(total_pop) <- 0
  update(total_pop) <- P_total

  # === Parameters ===
  S_0 <- parameter()
  E_0 <- parameter()
  I_0 <- parameter()
  R_0 <- parameter()
  V_0 <- parameter()
  vacc_rate <- parameter()

  dim(R0_time) <- parameter(rank = 1)
  dim(R0_value) <- parameter(rank = 1)
  dim(sp_time) <- parameter(rank = 1)
  dim(sp_value) <- parameter(rank = 1)
  dim(dP1_time) <- parameter(rank = 1)
  dim(dP1_value) <- parameter(rank = 1)
  dim(dP2_time) <- parameter(rank = 1)
  dim(dP2_value) <- parameter(rank = 1)
  R0_time <- parameter()
  R0_value <- parameter()
  sp_time <- parameter()
  sp_value <- parameter()
  dP1_time <- parameter()
  dP1_value <- parameter()
  dP2_time <- parameter()
  dP2_value <- parameter()
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

    ℹ Re-compiling odin.system1188989e

    ── R CMD INSTALL ───────────────────────────────────────────────────────────────
    * installing *source* package ‘odin.system1188989e’ ...
    ** this is package ‘odin.system1188989e’ version ‘0.0.1’
    ** using staged installation
    ** libs
    using C++ compiler: ‘Homebrew clang version 21.1.5’
    using SDK: ‘MacOSX15.5.sdk’
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c cpp11.cpp -o cpp11.o
    clang++ -arch arm64 -std=gnu++17 -I"/Library/Frameworks/R.framework/Resources/include" -DNDEBUG  -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/cpp11/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/dust2/include' -I'/Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/library/monty/include' -I/opt/R/arm64/include   -DHAVE_INLINE   -fPIC  -falign-functions=64 -Wall -g -O2  -Wall -pedantic  -c dust.cpp -o dust.o
    In file included from dust.cpp:342:
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
    dust.cpp:346:20: note: in instantiation of function template specialization 'dust2::r::dust2_discrete_alloc<odin_system>' requested here
      346 |   return dust2::r::dust2_discrete_alloc<odin_system>(r_pars, r_time, r_time_control, r_n_particles, r_n_groups, r_seed, r_deterministic, r_n_threads);
          |                    ^
    2 warnings generated.
    clang++ -arch arm64 -std=gnu++17 -dynamiclib -Wl,-headerpad_max_install_names -undefined dynamic_lookup -L/Library/Frameworks/R.framework/Resources/lib -L/opt/R/arm64/lib -o odin.system1188989e.so cpp11.o dust.o -F/Library/Frameworks/R.framework/.. -framework R
    installing to /private/var/folders/yh/30rj513j6mn1n7x556c2v4w80000gn/T/RtmpSz20ww/devtools_install_102f732d911af/00LOCK-dust_102f72e9c5107/00new/odin.system1188989e/libs
    ** checking absolute paths in shared objects and dynamic libraries
    * DONE (odin.system1188989e)

    ℹ Loading odin.system1188989e

## Parameter Setup

``` r
N_age <- 5
age_labels <- c("0-4", "5-14", "15-29", "30-49", "50+")

pop <- c(150000, 250000, 200000, 150000, 100000)
N_total <- sum(pop)

# Initial conditions
S_0 <- pop
E_0 <- rep(0, N_age)
I_0 <- rep(0, N_age)
R_0 <- rep(0, N_age)
V_0 <- rep(0, N_age)

# Pre-existing immunity
immun_frac <- c(0.05, 0.10, 0.15, 0.20, 0.30)
for (i in seq_len(N_age)) {
  R_0[i] <- round(pop[i] * immun_frac[i])
  S_0[i] <- S_0[i] - R_0[i]
}

# Pre-existing vaccination
vacc_frac <- c(0.30, 0.40, 0.20, 0.10, 0.05)
for (i in seq_len(N_age)) {
  V_0[i] <- round(pop[i] * vacc_frac[i])
  S_0[i] <- S_0[i] - V_0[i]
}

# Seed infections
I_0[3] <- 50
S_0[3] <- S_0[3] - 50

cat("S_0:", S_0, "\n")
```

    S_0: 97500 125000 129950 105000 65000 

``` r
cat("R_0:", R_0, "\n")
```

    R_0: 7500 25000 30000 30000 30000 

``` r
cat("V_0:", V_0, "\n")
```

    V_0: 45000 1e+05 40000 15000 5000 

### Time-varying parameters

``` r
n_years <- 10
t_end <- 365 * n_years

# R0: seasonal pattern
R0_time <- seq(0, t_end + 30, by = 30)
R0_value <- 3.5 + 1.5 * sin(2 * pi * R0_time / 365)

# Spillover FOI
sp_time <- seq(0, t_end + 30, by = 30)
sp_value <- 1e-6 + 5e-5 * pmax(0, sin(2 * pi * sp_time / 365 - pi / 3))^3

# Demographic rates
dP1_time <- c(0, t_end + 1)
dP1_value <- c(1, 1)
dP2_time <- c(0, t_end + 1)
dP2_value <- c(1, 1)

# Vaccination rates
vacc_rate <- c(0.001, 0.0005, 0.0003, 0.0002, 0.0001)

plot(R0_time[1:50], R0_value[1:50], type = "l", col = "steelblue", lwd = 2,
     xlab = "Day", ylab = "R0", main = "Time-varying R0 (first 4 years)")
```

![](20_yellowfever_files/figure-commonmark/unnamed-chunk-4-1.png)

### Assemble parameters

``` r
pars <- list(
  N_age = N_age,
  t_latent = 5,
  t_infectious = 5,
  vaccine_efficacy = 0.95,
  S_0 = S_0,
  E_0 = E_0,
  I_0 = I_0,
  R_0 = R_0,
  V_0 = V_0,
  vacc_rate = vacc_rate,
  R0_time = R0_time,
  R0_value = R0_value,
  sp_time = sp_time,
  sp_value = sp_value,
  dP1_time = dP1_time,
  dP1_value = dP1_value,
  dP2_time = dP2_time,
  dP2_value = dP2_value
)
```

## Simulation

``` r
n_particles <- 10
sim_times <- seq(0, t_end, by = 1)

sys <- dust_system_create(yf_seirv, pars, n_particles = n_particles,
                          dt = 1, seed = 42)
dust_system_set_state_initial(sys)
result <- dust_system_simulate(sys, sim_times)
cat("Result dimensions:", dim(result), "\n")
```

    Result dimensions: 33 10 3651 

### State layout

``` r
# State layout follows initial() declaration order:
# C[1:5] (zero_every declared first), S[6:10], E[11:15], I[16:20], R[21:25], V[26:30]
idx_C <- 1:N_age
idx_S <- (N_age + 1):(2 * N_age)
idx_E <- (2 * N_age + 1):(3 * N_age)
idx_I <- (3 * N_age + 1):(4 * N_age)
idx_R <- (4 * N_age + 1):(5 * N_age)
idx_V <- (5 * N_age + 1):(6 * N_age)
out_offset <- 6 * N_age
```

## Visualization

### SEIRV compartments (mean across particles)

``` r
S_total <- apply(apply(result[idx_S, , , drop = FALSE], c(2, 3), sum), 2, mean)
E_total <- apply(apply(result[idx_E, , , drop = FALSE], c(2, 3), sum), 2, mean)
I_total <- apply(apply(result[idx_I, , , drop = FALSE], c(2, 3), sum), 2, mean)
R_total <- apply(apply(result[idx_R, , , drop = FALSE], c(2, 3), sum), 2, mean)
V_total <- apply(apply(result[idx_V, , , drop = FALSE], c(2, 3), sum), 2, mean)

plot(sim_times, S_total, type = "l", col = "blue", lwd = 2,
     xlab = "Day", ylab = "Population",
     main = "Yellow Fever SEIRV Dynamics",
     ylim = c(0, max(S_total, V_total)))
lines(sim_times, E_total, col = "orange", lwd = 2)
lines(sim_times, I_total, col = "red", lwd = 2)
lines(sim_times, R_total, col = "green", lwd = 2)
lines(sim_times, V_total, col = "purple", lwd = 2)
legend("right", legend = c("S", "E", "I", "R", "V"),
       col = c("blue", "orange", "red", "green", "purple"), lwd = 2)
```

![](20_yellowfever_files/figure-commonmark/unnamed-chunk-8-1.png)

### Infectious by age group

``` r
age_cols <- c("blue", "red", "green", "orange", "purple")
plot(NULL, xlim = range(sim_times), ylim = c(0, max(result[idx_I, , ])),
     xlab = "Day", ylab = "Mean Infectious",
     main = "Infectious by Age Group")
for (a in seq_len(N_age)) {
  I_a <- apply(result[idx_I[a], , , drop = FALSE], 3, mean)
  lines(sim_times, I_a, col = age_cols[a], lwd = 2)
}
legend("topright", legend = age_labels, col = age_cols, lwd = 2)
```

![](20_yellowfever_files/figure-commonmark/unnamed-chunk-9-1.png)

### Vaccination coverage

``` r
plot(NULL, xlim = range(sim_times), ylim = c(0, 100),
     xlab = "Day", ylab = "Mean Vaccinated (%)",
     main = "Vaccination Coverage by Age Group")
for (a in seq_len(N_age)) {
  V_a <- apply(result[idx_V[a], , , drop = FALSE], 3, mean)
  coverage <- V_a / pop[a] * 100
  lines(sim_times, coverage, col = age_cols[a], lwd = 2)
}
legend("topleft", legend = age_labels, col = age_cols, lwd = 2)
```

![](20_yellowfever_files/figure-commonmark/unnamed-chunk-10-1.png)

### Force of infection

``` r
FOI <- apply(result[out_offset + 1, , , drop = FALSE], 3, mean)
plot(sim_times, FOI, type = "l", col = "darkred", lwd = 1.5,
     xlab = "Day", ylab = "FOI",
     main = "Total Force of Infection (mean across particles)")
```

![](20_yellowfever_files/figure-commonmark/unnamed-chunk-11-1.png)

### Cumulative cases

``` r
for (a in seq_len(N_age)) {
  daily <- apply(result[idx_C[a], , , drop = FALSE], 3, mean)
  cum <- cumsum(daily)
  if (a == 1) {
    plot(sim_times, cum, type = "l", col = age_cols[a], lwd = 2,
         xlab = "Day", ylab = "Cumulative Cases",
         main = "Cumulative Cases by Age Group",
         ylim = c(0, max(cum) * N_age))
  } else {
    lines(sim_times, cum, col = age_cols[a], lwd = 2)
  }
}
legend("topleft", legend = age_labels, col = age_cols, lwd = 2)
```

![](20_yellowfever_files/figure-commonmark/unnamed-chunk-12-1.png)

## Population Conservation Check

``` r
pop_check <- rep(0, length(sim_times))
for (i in seq_len(n_particles)) {
  pop_i <- colSums(result[idx_S, i, ]) + colSums(result[idx_E, i, ]) +
           colSums(result[idx_I, i, ]) + colSums(result[idx_R, i, ]) +
           colSums(result[idx_V, i, ])
  pop_check <- pop_check + pop_i
}
pop_check <- pop_check / n_particles

plot(sim_times, pop_check, type = "l", col = "steelblue", lwd = 2,
     xlab = "Day", ylab = "Total Population",
     main = "Population Conservation Check")
abline(h = N_total, col = "black", lwd = 2, lty = 2)
```

![](20_yellowfever_files/figure-commonmark/unnamed-chunk-13-1.png)

``` r
cat("Population at t=0:", pop_check[1], "\n")
```

    Population at t=0: 790000 

``` r
cat("Population at t=end:", pop_check[length(pop_check)], "\n")
```

    Population at t=end: 789985.1 

``` r
cat("Relative change:", round(abs(pop_check[length(pop_check)] - N_total) /
    N_total * 100, 4), "%\n")
```

    Relative change: 7.0606 %

## Scenario: No Vaccination

``` r
pars_novacc <- pars
pars_novacc$vacc_rate <- rep(0, N_age)
pars_novacc$V_0 <- rep(0, N_age)
pars_novacc$S_0 <- pop - R_0 - I_0

sys_nv <- dust_system_create(yf_seirv, pars_novacc, n_particles = n_particles,
                             dt = 1, seed = 42)
dust_system_set_state_initial(sys_nv)
result_nv <- dust_system_simulate(sys_nv, sim_times)

I_vacc <- apply(apply(result[idx_I, , , drop = FALSE], c(2, 3), sum), 2, mean)
I_novacc <- apply(apply(result_nv[idx_I, , , drop = FALSE], c(2, 3), sum), 2, mean)

plot(sim_times, I_novacc, type = "l", col = "red", lwd = 2,
     xlab = "Day", ylab = "Mean Total Infectious",
     main = "Impact of Vaccination on Yellow Fever")
lines(sim_times, I_vacc, col = "blue", lwd = 2)
legend("topright", legend = c("No vaccination", "With vaccination"),
       col = c("red", "blue"), lwd = 2)
```

![](20_yellowfever_files/figure-commonmark/unnamed-chunk-14-1.png)
