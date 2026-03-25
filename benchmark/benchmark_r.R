#!/usr/bin/env Rscript
# R benchmarks using odin2/dust2/monty — comparable to benchmark_julia.jl
#
# Run: Rscript benchmark/benchmark_r.R

library(odin2)
library(dust2)
library(monty)
library(microbenchmark)

cat(strrep("=", 70), "\n")
cat("odin2/dust2/monty R Benchmarks\n")
cat(strrep("=", 70), "\n")

run_bench <- function(name, expr, times = 50) {
  mb <- microbenchmark(eval(expr), times = times)
  med_ms <- median(mb$time) / 1e6
  min_ms <- min(mb$time) / 1e6
  max_ms <- max(mb$time) / 1e6
  cat(sprintf("  %-35s %10.3f ms (min %.3f, max %.3f)\n", name, med_ms, min_ms, max_ms))
  list(name = name, median_ms = med_ms, min_ms = min_ms, max_ms = max_ms)
}

# ── 1. ODE SIR simulation ──────────────────────────────────────

sir_ode <- odin({
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

pars_ode <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
times_ode <- seq(0, 365, by = 1)

cat("\n1. ODE SIR — simulate 365 days, 1 particle\n")
r1 <- run_bench("ODE SIR (365d, 1 part.)", quote({
  sys <- dust_system_create(sir_ode, pars_ode, ode_control = dust_ode_control())
  dust_system_set_state_initial(sys)
  dust_system_simulate(sys, times_ode)
}))

# ── 2. Stochastic SIR simulation ───────────────────────────────

sir_stoch <- odin({
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
  beta <- parameter(0.5)
  gamma <- parameter(0.1)
  I0 <- parameter(10)
  N <- parameter(1000)
})

pars_stoch <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
times_stoch <- seq(0, 365, by = 1)

cat("\n2. Stochastic SIR — 365 days, 100 particles\n")
r2 <- run_bench("Stochastic SIR (365d, 100p)", quote({
  sys <- dust_system_create(sir_stoch, pars_stoch, n_particles = 100, dt = 1, seed = 1)
  dust_system_set_state_initial(sys)
  dust_system_simulate(sys, times_stoch)
}))

# ── 3. Age-structured ODE SIR ──────────────────────────────────

sir_age <- odin({
  n_age <- parameter(10)
  dim(S, I, R) <- n_age
  dim(beta_vec, S0, I0_vec) <- n_age

  deriv(S[]) <- -beta_vec[i] * S[i] * total_I / N
  deriv(I[]) <- beta_vec[i] * S[i] * total_I / N - gamma * I[i]
  deriv(R[]) <- gamma * I[i]

  total_I <- sum(I)

  initial(S[]) <- S0[i]
  initial(I[]) <- I0_vec[i]
  initial(R[]) <- 0

  S0 <- parameter()
  I0_vec <- parameter()
  beta_vec <- parameter()
  gamma <- parameter(0.1)
  N <- parameter(10000)
})

pars_age <- list(
  n_age = 10,
  S0 = rep(990, 10),
  I0_vec = rep(10, 10),
  beta_vec = seq(0.2, 0.6, length.out = 10),
  gamma = 0.1,
  N = 10000
)
times_age <- seq(0, 365, by = 1)

cat("\n3. Age-structured ODE SIR (10 groups) — 365 days\n")
r3 <- run_bench("Age-struct ODE (10grp, 365d)", quote({
  sys <- dust_system_create(sir_age, pars_age, ode_control = dust_ode_control())
  dust_system_set_state_initial(sys)
  dust_system_simulate(sys, times_age)
}))

# ── 4. Particle filter ─────────────────────────────────────────

sir_pf <- odin({
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

# Generate data
pars_pf <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
times_pf <- seq(0, 100, by = 1)
sys_pf <- dust_system_create(sir_pf, pars_pf, dt = 1, seed = 1)
dust_system_set_state_initial(sys_pf)
sim_data <- dust_system_simulate(sys_pf, times_pf)
# sim_data is (n_state, n_times) for 1 particle; incidence is state 4
obs <- round(sim_data[4, -1])
data_pf <- data.frame(time = times_pf[-1], cases = obs)

filter <- dust_filter_create(sir_pf, time_start = 0, data = data_pf,
                             n_particles = 200, seed = 42)

cat("\n4. Particle filter — 100 days, 200 particles\n")
r4 <- run_bench("Particle filter (100d, 200p)", quote({
  dust_likelihood_run(filter, pars_pf)
}))

# ── 5. MCMC sampling (short chain) ─────────────────────────────

packer <- monty_packer(c("beta", "gamma"),
                       fixed = list(I0 = 10, N = 1000))
likelihood <- dust_likelihood_monty(filter, packer)
prior <- monty_dsl({
  beta ~ Gamma(shape = 2, rate = 4)
  gamma ~ Gamma(shape = 2, rate = 20)
})
posterior <- likelihood + prior
sampler <- monty_sampler_random_walk(matrix(c(0.005, 0, 0, 0.001), 2, 2))

cat("\n5. MCMC — 500 iterations, RW sampler, particle filter likelihood\n")
r5 <- run_bench("MCMC 500 iter (PF + RW)", quote({
  monty_sample(posterior, sampler, 500, initial = c(0.4, 0.08))
}), times = 5)
# ── Summary table ───────────────────────────────────────────────

cat("\n", strrep("=", 70), "\n")
cat("Summary (median times)\n")
cat(strrep("=", 70), "\n")

results <- list(r1, r2, r3, r4, r5)
df <- data.frame(
  benchmark = c("ode_sir", "stoch_sir", "age_sir", "particle_filter", "mcmc_500"),
  median_ms = sapply(results, `[[`, "median_ms"),
  min_ms = sapply(results, `[[`, "min_ms"),
  max_ms = sapply(results, `[[`, "max_ms")
)
write.csv(df, "benchmark/results_r.csv", row.names = FALSE)
cat("Results saved to benchmark/results_r.csv\n")
