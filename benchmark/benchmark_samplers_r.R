#!/usr/bin/env Rscript
# R sampler ESS benchmarks — companion to benchmark_samplers_julia.jl
#
# Compares R's RW MCMC (deterministic unfilter) and particle filter MCMC
# so Julia's NUTS / HMC ESS/sec can be compared against R's baseline.
#
# Run: Rscript benchmark/benchmark_samplers_r.R

library(odin2)
library(dust2)
library(monty)
library(posterior)  # for ESS computation

cat(strrep("=", 72), "\n")
cat("odin2/dust2/monty R Sampler ESS Benchmarks\n")
cat(strrep("=", 72), "\n")

# ── ODE SIR model ───────────────────────────────────────────

sir_ode <- odin({
  deriv(S) <- -beta * S * I / N
  deriv(I) <- beta * S * I / N - gamma * I
  deriv(R) <- gamma * I
  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0
  N <- parameter(1000)
  I0 <- parameter(10)
  beta <- parameter(0.5)
  gamma <- parameter(0.1)
  cases_lambda <- if (I > 0) rho * I else 1e-10
  cases <- data()
  cases ~ Poisson(cases_lambda)
  rho <- parameter(0.3)
})

# ── Generate synthetic data (same seed logic as Julia) ──────

true_pars <- list(beta = 0.5, gamma = 0.1, rho = 0.3, I0 = 10, N = 1000)
sys <- dust_system_create(sir_ode, true_pars, seed = 42)
dust_system_set_state_initial(sys)
times <- 1:50
sim <- dust_system_simulate(sys, times)
# State row 2 is I; generate Poisson-like data
obs <- pmax(1, round(sim[2, ] * 0.3))
data_ode <- data.frame(time = times, cases = obs)

# ══════════════════════════════════════════════════════════════════
# PART 1: ODE model — RW MCMC (R's only option without adjoint)
# ══════════════════════════════════════════════════════════════════

cat("\n", strrep("-", 72), "\n")
cat("PART 1: ODE SIR — RW MCMC (R)\n")
cat(strrep("-", 72), "\n")

packer <- monty_packer(c("beta", "gamma", "rho"),
                       fixed = list(I0 = 10, N = 1000))
filter <- dust_unfilter_create(sir_ode, time_start = 0, data = data_ode)
likelihood <- dust_likelihood_monty(filter, packer)

prior <- monty_dsl({
  beta ~ Exponential(mean = 1)
  gamma ~ Exponential(mean = 1)
  rho ~ Beta(2, 5)
})
posterior <- likelihood + prior

vcv <- diag(c(0.002, 0.0005, 0.003))
sampler <- monty_sampler_random_walk(vcv)

n_steps <- 2000L
n_burnin <- 500L
n_chains <- 4L

initial <- matrix(c(0.4, 0.08, 0.25,
                     0.45, 0.09, 0.28,
                     0.55, 0.11, 0.32,
                     0.6, 0.12, 0.35), nrow = 3, ncol = 4)

# Warmup
monty_sample(posterior, sampler, 50, initial = initial[, 1, drop = FALSE])

cat("\n  Running RW MCMC (", n_steps, " steps, ", n_chains, " chains)...\n")
t0 <- proc.time()
samples_rw <- monty_sample(posterior, sampler, n_steps,
                           initial = initial, n_chains = n_chains,
                           burnin = n_burnin)
elapsed_rw <- (proc.time() - t0)["elapsed"]

# Extract posterior draws and compute ESS using 'posterior' package
# monty_sample returns a list with $pars (n_pars × n_steps × n_chains)
draws <- samples_rw$pars
param_names <- c("beta", "gamma", "rho")
n_samples <- dim(draws)[2]

# Convert to posterior::draws_array (iterations × chains × variables)
draws_arr <- array(NA, dim = c(n_samples, n_chains, 3),
                   dimnames = list(iteration = NULL,
                                   chain = paste0("chain:", 1:n_chains),
                                   variable = param_names))
for (p in 1:3) {
  for (ch in 1:n_chains) {
    draws_arr[, ch, p] <- draws[p, , ch]
  }
}

da <- posterior::as_draws_array(draws_arr)
ess_vals <- posterior::ess_bulk(da)
rhat_vals <- posterior::rhat(da)

cat(sprintf("  Time: %.2f s\n", elapsed_rw))
for (p in param_names) {
  cat(sprintf("    %s: ESS = %.1f, ESS/sec = %.1f, rhat = %.4f\n",
              p, ess_vals[[p]], ess_vals[[p]] / elapsed_rw, rhat_vals[[p]]))
}

# ══════════════════════════════════════════════════════════════════
# PART 2: Stochastic model — RW MCMC + particle filter
# ══════════════════════════════════════════════════════════════════

cat("\n", strrep("-", 72), "\n")
cat("PART 2: Stochastic SIR — RW MCMC + Particle Filter (R)\n")
cat(strrep("-", 72), "\n")

sir_stoch <- odin({
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

stoch_pars <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
stoch_sys <- dust_system_create(sir_stoch, stoch_pars, dt = 1, seed = 1)
dust_system_set_state_initial(stoch_sys)
stoch_sim <- dust_system_simulate(stoch_sys, 0:100)
obs_stoch <- round(stoch_sim[4, -1])
stoch_data <- data.frame(time = 1:100, cases = obs_stoch)

packer_stoch <- monty_packer(c("beta", "gamma"), fixed = list(I0 = 10, N = 1000))
stoch_prior <- monty_dsl({
  beta ~ Gamma(shape = 2, rate = 4)
  gamma ~ Gamma(shape = 2, rate = 20)
})
stoch_vcv <- matrix(c(0.005, 0, 0, 0.001), 2, 2)
stoch_sampler <- monty_sampler_random_walk(stoch_vcv)
stoch_init <- matrix(c(0.4, 0.08, 0.45, 0.09, 0.55, 0.11, 0.6, 0.12), nrow = 2)

for (np in c(200, 500)) {
  f <- dust_filter_create(sir_stoch, time_start = 0, data = stoch_data,
                          n_particles = np, seed = 42)
  ll <- dust_likelihood_monty(f, packer_stoch)
  post <- ll + stoch_prior

  # warmup
  monty_sample(post, stoch_sampler, 50, initial = stoch_init[, 1, drop = FALSE])

  cat(sprintf("\n  RW + PF (%d particles)...\n", np))
  t0 <- proc.time()
  stoch_samples <- monty_sample(post, stoch_sampler, 2000,
                                initial = stoch_init, n_chains = 4, burnin = 500)
  elapsed <- (proc.time() - t0)["elapsed"]

  stoch_draws <- stoch_samples$pars
  n_s <- dim(stoch_draws)[2]
  stoch_arr <- array(NA, dim = c(n_s, 4, 2),
                     dimnames = list(NULL, paste0("chain:", 1:4), c("beta", "gamma")))
  for (p in 1:2) {
    for (ch in 1:4) {
      stoch_arr[, ch, p] <- stoch_draws[p, , ch]
    }
  }
  da_stoch <- posterior::as_draws_array(stoch_arr)
  ess_stoch <- posterior::ess_bulk(da_stoch)
  rhat_stoch <- posterior::rhat(da_stoch)

  cat(sprintf("    Time: %.2f s | ESS: [%.1f, %.1f] | ESS/s: [%.1f, %.1f] | rhat: [%.4f, %.4f]\n",
              elapsed, ess_stoch[["beta"]], ess_stoch[["gamma"]],
              ess_stoch[["beta"]] / elapsed, ess_stoch[["gamma"]] / elapsed,
              rhat_stoch[["beta"]], rhat_stoch[["gamma"]]))
}

# ── Save results ────────────────────────────────────────────

results_df <- data.frame(
  sampler = "RW",
  time_sec = elapsed_rw,
  param = param_names,
  ess = as.numeric(ess_vals[param_names]),
  ess_per_sec = as.numeric(ess_vals[param_names]) / elapsed_rw,
  rhat = as.numeric(rhat_vals[param_names])
)

write.csv(results_df, "benchmark/results_sampler_ess_r.csv", row.names = FALSE)
cat("\nResults saved to benchmark/results_sampler_ess_r.csv\n")
