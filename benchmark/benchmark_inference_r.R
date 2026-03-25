#!/usr/bin/env Rscript
# R inference benchmarks — particle filter & MCMC
# Companion to benchmark_inference_julia.jl
#
# Run: Rscript benchmark/benchmark_inference_r.R

library(odin2)
library(dust2)
library(monty)
library(microbenchmark)

cat(strrep("=", 72), "\n")
cat("odin2/dust2/monty R Inference Benchmarks\n")
cat(strrep("=", 72), "\n")

# ── Model definition ────────────────────────────────────────────

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

# ── Generate synthetic data ─────────────────────────────────────

true_pars <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
n_days <- 100
times <- seq(0, n_days, by = 1)
sys <- dust_system_create(sir, true_pars, dt = 1, seed = 1)
dust_system_set_state_initial(sys)
sim <- dust_system_simulate(sys, times)
obs <- round(sim[4, -1])
data_pf <- data.frame(time = seq_len(n_days), cases = obs)

# ══════════════════════════════════════════════════════════════════
# PART 1: PARTICLE FILTER — vary n_particles
# ══════════════════════════════════════════════════════════════════

cat("\n", strrep("-", 72), "\n")
cat("PART 1: Particle Filter  (100 days, varying n_particles)\n")
cat(strrep("-", 72), "\n")

pf_results <- data.frame(
  n_particles = integer(),
  median_ms = numeric(),
  min_ms = numeric(),
  max_ms = numeric()
)

for (np in c(50, 100, 200, 500, 1000, 2000)) {
  filter <- dust_filter_create(sir, time_start = 0, data = data_pf,
                               n_particles = np, seed = 42)
  # warmup
  dust_likelihood_run(filter, true_pars)
  dust_likelihood_run(filter, true_pars)

  mb <- microbenchmark(dust_likelihood_run(filter, true_pars), times = 30)
  med <- median(mb$time) / 1e6
  mn <- min(mb$time) / 1e6
  mx <- max(mb$time) / 1e6
  cat(sprintf("  n_particles=%5d  →  %8.2f ms  (min %7.2f, max %8.2f)\n",
              np, med, mn, mx))
  pf_results <- rbind(pf_results,
    data.frame(n_particles = np, median_ms = med, min_ms = mn, max_ms = mx))
}

# ── Verify filter correctness ───────────────────────────────────

filter_check <- dust_filter_create(sir, time_start = 0, data = data_pf,
                                   n_particles = 1000, seed = 42)
lls <- replicate(20, dust_likelihood_run(filter_check, true_pars))
cat(sprintf("\n  Log-likelihood check (1000 particles, 20 runs):\n"))
cat(sprintf("    mean = %.2f  ±  %.2f\n", mean(lls), sd(lls)))

# ══════════════════════════════════════════════════════════════════
# PART 2: MCMC — vary n_steps, n_chains
# ══════════════════════════════════════════════════════════════════

cat("\n", strrep("-", 72), "\n")
cat("PART 2: MCMC Sampling  (RW sampler + particle filter likelihood)\n")
cat(strrep("-", 72), "\n")

packer <- monty_packer(c("beta", "gamma"), fixed = list(I0 = 10, N = 1000))
filter_mcmc <- dust_filter_create(sir, time_start = 0, data = data_pf,
                                  n_particles = 200, seed = 42)
likelihood <- dust_likelihood_monty(filter_mcmc, packer)
prior <- monty_dsl({
  beta ~ Gamma(shape = 2, rate = 4)
  gamma ~ Gamma(shape = 2, rate = 20)
})
posterior <- likelihood + prior
sampler <- monty_sampler_random_walk(matrix(c(0.005, 0, 0, 0.001), 2, 2))

# warmup
monty_sample(posterior, sampler, 10, initial = c(0.4, 0.08))

mcmc_results <- data.frame(
  config = character(),
  n_steps = integer(),
  n_chains = integer(),
  median_ms = numeric(),
  min_ms = numeric(),
  max_ms = numeric(),
  stringsAsFactors = FALSE
)

# 2a: Vary n_steps (1 chain)
cat("\n  2a: Vary n_steps (1 chain, 200 particles)\n")
for (ns in c(100, 500, 1000, 2000)) {
  mb <- microbenchmark(
    monty_sample(posterior, sampler, ns, initial = c(0.4, 0.08)),
    times = 5
  )
  med <- median(mb$time) / 1e6
  mn <- min(mb$time) / 1e6
  mx <- max(mb$time) / 1e6
  cat(sprintf("    n_steps=%5d  →  %10.1f ms  (min %9.1f, max %10.1f)\n",
              ns, med, mn, mx))
  mcmc_results <- rbind(mcmc_results,
    data.frame(config = paste0("steps_", ns, "_1chain"),
               n_steps = ns, n_chains = 1L,
               median_ms = med, min_ms = mn, max_ms = mx))
}

# 2b: Vary n_chains (500 steps each)
cat("\n  2b: Vary n_chains (500 steps, 200 particles)\n")
for (nc in c(1, 2, 4)) {
  init <- matrix(rep(c(0.4, 0.08), nc), 2, nc)
  mb <- microbenchmark(
    monty_sample(posterior, sampler, 500, initial = init, n_chains = nc),
    times = 5
  )
  med <- median(mb$time) / 1e6
  mn <- min(mb$time) / 1e6
  mx <- max(mb$time) / 1e6
  cat(sprintf("    n_chains=%d  →  %10.1f ms  (min %9.1f, max %10.1f)\n",
              nc, med, mn, mx))
  mcmc_results <- rbind(mcmc_results,
    data.frame(config = paste0("500steps_", nc, "chain"),
               n_steps = 500L, n_chains = nc,
               median_ms = med, min_ms = mn, max_ms = mx))
}

# 2c: Effect of n_particles on MCMC
cat("\n  2c: Vary n_particles in MCMC (500 steps, 1 chain)\n")
for (np in c(50, 200, 500, 1000)) {
  f <- dust_filter_create(sir, time_start = 0, data = data_pf,
                          n_particles = np, seed = 42)
  ll <- dust_likelihood_monty(f, packer)
  post <- ll + prior
  # warmup
  monty_sample(post, sampler, 10, initial = c(0.4, 0.08))
  mb <- microbenchmark(
    monty_sample(post, sampler, 500, initial = c(0.4, 0.08)),
    times = 5
  )
  med <- median(mb$time) / 1e6
  cat(sprintf("    n_particles=%5d  →  %10.1f ms\n", np, med))
  mcmc_results <- rbind(mcmc_results,
    data.frame(config = paste0("500steps_", np, "part"),
               n_steps = 500L, n_chains = 1L,
               median_ms = med,
               min_ms = min(mb$time) / 1e6,
               max_ms = max(mb$time) / 1e6))
}

# ── Summary ──────────────────────────────────────────────────────

cat("\n", strrep("=", 72), "\n")
cat("SUMMARY\n")
cat(strrep("=", 72), "\n")

cat("\nParticle Filter (100 days):\n")
cat(sprintf("  %-20s  %10s  %10s\n", "n_particles", "median_ms", "ms/particle"))
for (i in seq_len(nrow(pf_results))) {
  r <- pf_results[i, ]
  cat(sprintf("  %-20d  %10.2f  %10.4f\n", r$n_particles, r$median_ms,
              r$median_ms / r$n_particles))
}

cat("\nMCMC:\n")
cat(sprintf("  %-25s  %10s\n", "config", "median_ms"))
for (i in seq_len(nrow(mcmc_results))) {
  r <- mcmc_results[i, ]
  cat(sprintf("  %-25s  %10.1f\n", r$config, r$median_ms))
}

# Save results
all_results <- rbind(
  data.frame(type = "pf",
             config = paste0("np_", pf_results$n_particles),
             median_ms = pf_results$median_ms,
             min_ms = pf_results$min_ms,
             max_ms = pf_results$max_ms),
  data.frame(type = "mcmc",
             config = mcmc_results$config,
             median_ms = mcmc_results$median_ms,
             min_ms = mcmc_results$min_ms,
             max_ms = mcmc_results$max_ms)
)
write.csv(all_results, "benchmark/results_inference_r.csv", row.names = FALSE)
cat("\nResults saved to benchmark/results_inference_r.csv\n")
