#!/usr/bin/env Rscript
# Benchmark: R odin2/dust2/monty HMC/NUTS on SIR ODE
# Companion to benchmark_hmc_symbolic.jl

library(odin2)
library(dust2)
library(monty)

# ──────────────────────────────────────────────────────────────
# 1. Define SIR ODE model
# ──────────────────────────────────────────────────────────────

sir <- odin({
  deriv(S) <- -beta * S * I / N
  deriv(I) <- beta * S * I / N - gamma * I
  deriv(R) <- gamma * I
  initial(S) <- N - I0
  initial(I) <- I0
  initial(R) <- 0
  N <- parameter(1000)
  I0 <- parameter(10)
  beta <- parameter(0.5, differentiate = TRUE)
  gamma <- parameter(0.1, differentiate = TRUE)
  rho <- parameter(0.3, differentiate = TRUE)
  cases_lambda <- if (I > 0) rho * I else 1e-10
  cases <- data()
  cases ~ Poisson(cases_lambda)
})

# ──────────────────────────────────────────────────────────────
# 2. Load same data as Julia benchmark
# ──────────────────────────────────────────────────────────────

data_file <- file.path(dirname(commandArgs(trailingOnly = FALSE)[
  grep("--file=", commandArgs(trailingOnly = FALSE))
] |> sub("--file=", "", x = _) |> (\(x) if (length(x)) x else ".")()),
  "benchmark_hmc_data.csv")
if (!file.exists(data_file)) {
  data_file <- "benchmark/benchmark_hmc_data.csv"
}
if (!file.exists(data_file)) {
  # Generate data if Julia hasn't been run yet
  set.seed(42)
  true_pars <- list(beta = 0.5, gamma = 0.1, rho = 0.3, I0 = 10, N = 1000)
  sys <- dust_system_create(sir, true_pars, seed = 42L)
  dust_system_set_state_initial(sys)
  times <- 1:50
  sim <- dust_system_simulate(sys, times)
  I_traj <- sim["I", 1, ]
  set.seed(123)
  obs <- pmax(1, rpois(length(times), 0.3 * pmax(1e-10, I_traj)))
  data_df <- data.frame(time = times, cases = obs)
} else {
  data_df <- read.csv(data_file)
}

cat(sprintf("Data: %d time points, mean cases = %.1f\n",
            nrow(data_df), mean(data_df$cases)))

# ──────────────────────────────────────────────────────────────
# 3. Setup posterior
# ──────────────────────────────────────────────────────────────

packer <- monty_packer(c("beta", "gamma", "rho"),
                       fixed = list(I0 = 10, N = 1000))

uf <- dust_unfilter_create(sir, time_start = 0, data = data_df)
ll <- dust_likelihood_monty(uf, packer)

cat(sprintf("Likelihood has gradient: %s\n", ll$properties$has_gradient))

prior <- monty_dsl({
  beta ~ Exponential(mean = 1)
  gamma ~ Exponential(mean = 1)
  rho ~ Beta(2, 5)
})

posterior <- ll + prior

# ──────────────────────────────────────────────────────────────
# 4. Benchmark settings
# ──────────────────────────────────────────────────────────────

n_steps <- 2000
n_chains <- 4
n_burnin <- 500

initial <- matrix(c(0.4, 0.08, 0.25,
                     0.45, 0.09, 0.28,
                     0.55, 0.11, 0.32,
                     0.6, 0.12, 0.35),
                  nrow = 3, ncol = 4)

results <- list()

compute_ess <- function(samples, n_burnin) {
  # Use posterior package if available, else batch means
  post <- samples$pars[, (n_burnin + 1):n_steps, , drop = FALSE]
  n_params <- dim(post)[1]
  ess <- numeric(n_params)
  for (p in seq_len(n_params)) {
    vals <- as.vector(post[p, , ])
    n <- length(vals)
    v <- var(vals)
    if (v < 1e-20) {
      ess[p] <- 0
      next
    }
    batch_size <- max(1, floor(sqrt(n)))
    n_batches <- floor(n / batch_size)
    batch_means <- vapply(seq_len(n_batches), function(b) {
      idx <- ((b - 1) * batch_size + 1):(b * batch_size)
      mean(vals[idx])
    }, numeric(1))
    var_bm <- var(batch_means) * batch_size
    ess[p] <- n * v / max(var_bm, 1e-20)
  }
  return(ess)
}

# ──────────────────────────────────────────────────────────────
# 5. Run benchmarks
# ──────────────────────────────────────────────────────────────

# ── Random Walk (baseline) ──
cat("\n=== R Random Walk (baseline) ===\n")
vcv <- diag(c(0.002, 0.0005, 0.003))
rw <- monty_sampler_random_walk(vcv)

t_rw <- system.time({
  res_rw <- monty_sample(posterior, rw, n_steps,
                         n_chains = n_chains, initial = initial)
})["elapsed"]
ess_rw <- compute_ess(res_rw, n_burnin)
cat(sprintf("  Time: %.2fs\n", t_rw))
cat(sprintf("  ESS: beta=%.1f, gamma=%.1f, rho=%.1f\n",
            ess_rw[1], ess_rw[2], ess_rw[3]))
cat(sprintf("  ESS/s: %.1f, %.1f, %.1f\n", ess_rw[1]/t_rw, ess_rw[2]/t_rw, ess_rw[3]/t_rw))
results$R_RW <- list(time = t_rw, ess = ess_rw)

# ── HMC ──
cat("\n=== R HMC ===\n")
hmc <- monty_sampler_hmc(epsilon = 0.005, n_integration_steps = 20)

t_hmc <- system.time({
  res_hmc <- monty_sample(posterior, hmc, n_steps,
                          n_chains = n_chains, initial = initial)
})["elapsed"]
ess_hmc <- compute_ess(res_hmc, n_burnin)
cat(sprintf("  Time: %.2fs\n", t_hmc))
cat(sprintf("  ESS: beta=%.1f, gamma=%.1f, rho=%.1f\n",
            ess_hmc[1], ess_hmc[2], ess_hmc[3]))
cat(sprintf("  ESS/s: %.1f, %.1f, %.1f\n", ess_hmc[1]/t_hmc, ess_hmc[2]/t_hmc, ess_hmc[3]/t_hmc))
results$R_HMC <- list(time = t_hmc, ess = ess_hmc)

# ── NUTS ──
cat("\n=== R NUTS ===\n")
tryCatch({
  nuts <- monty_sampler_nuts(max_treedepth = 10, n_warmup = n_burnin,
                             target_accept = 0.8)
  t_nuts <- system.time({
    res_nuts <- monty_sample(posterior, nuts, n_steps,
                             n_chains = n_chains, initial = initial)
  })["elapsed"]
  ess_nuts <- compute_ess(res_nuts, n_burnin)
  cat(sprintf("  Time: %.2fs\n", t_nuts))
  cat(sprintf("  ESS: beta=%.1f, gamma=%.1f, rho=%.1f\n",
              ess_nuts[1], ess_nuts[2], ess_nuts[3]))
  cat(sprintf("  ESS/s: %.1f, %.1f, %.1f\n", ess_nuts[1]/t_nuts, ess_nuts[2]/t_nuts, ess_nuts[3]/t_nuts))
  results$R_NUTS <- list(time = t_nuts, ess = ess_nuts)
}, error = function(e) {
  cat(sprintf("  NUTS failed: %s\n", conditionMessage(e)))
})

# ──────────────────────────────────────────────────────────────
# 6. Summary
# ──────────────────────────────────────────────────────────────

cat("\n")
cat(strrep("=", 80), "\n")
cat(sprintf("BENCHMARK SUMMARY: SIR ODE — %d steps × %d chains\n",
            n_steps, n_chains))
cat(strrep("=", 80), "\n")
cat(sprintf("%-25s %8s %8s %8s %8s %8s %8s\n",
            "Sampler", "Time(s)", "ESS_β", "ESS_γ", "ESS_ρ",
            "ESS/s_β", "ESS/s_γ"))
cat(strrep("-", 80), "\n")

for (name in names(results)) {
  r <- results[[name]]
  cat(sprintf("%-25s %8.2f %8.1f %8.1f %8.1f %8.1f %8.1f\n",
              name, r$time, r$ess[1], r$ess[2], r$ess[3],
              r$ess[1] / r$time, r$ess[2] / r$time))
}

# ──────────────────────────────────────────────────────────────
# 7. Save results
# ──────────────────────────────────────────────────────────────

outfile <- file.path(dirname(data_file), "results_hmc_benchmark_r.csv")
out <- do.call(rbind, lapply(names(results), function(name) {
  r <- results[[name]]
  data.frame(
    sampler = name,
    time_sec = r$time,
    param = c("beta", "gamma", "rho"),
    ess = r$ess,
    ess_per_sec = r$ess / r$time
  )
}))
write.csv(out, outfile, row.names = FALSE)
cat(sprintf("\nR results saved to: %s\n", outfile))

# Posterior means
cat("\nPosterior means (R HMC):\n")
post_hmc <- res_hmc$pars[, (n_burnin + 1):n_steps, ]
for (p in 1:3) {
  vals <- as.vector(post_hmc[p, , ])
  cat(sprintf("  %s: mean=%.4f, std=%.4f (true=%.1f)\n",
              c("beta", "gamma", "rho")[p],
              mean(vals), sd(vals),
              c(0.5, 0.1, 0.3)[p]))
}
