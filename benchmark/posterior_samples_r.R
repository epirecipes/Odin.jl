#!/usr/bin/env Rscript
# Save posterior samples from R MCMC for cross-language ECDF comparison
#
# Run: Rscript benchmark/posterior_samples_r.R

library(odin2)
library(dust2)
library(monty)

cat("Generating R posterior samples for ECDF comparison...\n")

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

# Generate synthetic data with fixed seed
true_pars <- list(beta = 0.5, gamma = 0.1, I0 = 10, N = 1000)
sys <- dust_system_create(sir, true_pars, dt = 1, seed = 1)
dust_system_set_state_initial(sys)
times <- seq(0, 100, by = 1)
sim <- dust_system_simulate(sys, times)
obs <- round(sim[4, -1])
data_pf <- data.frame(time = seq_len(100), cases = obs)

# Save the data so Julia uses the exact same observations
write.csv(data_pf, "benchmark/shared_data.csv", row.names = FALSE)
cat("  Saved shared_data.csv (100 obs)\n")

# Run MCMC with many particles and long chain for good posterior
filter <- dust_filter_create(sir, time_start = 0, data = data_pf,
                             n_particles = 500, seed = 42)
packer <- monty_packer(c("beta", "gamma"), fixed = list(I0 = 10, N = 1000))
likelihood <- dust_likelihood_monty(filter, packer)
prior <- monty_dsl({
  beta ~ Gamma(shape = 2, rate = 4)
  gamma ~ Gamma(shape = 2, rate = 20)
})
posterior <- likelihood + prior
sampler <- monty_sampler_random_walk(matrix(c(0.005, 0, 0, 0.001), 2, 2))

cat("  Running MCMC: 5000 steps × 4 chains, 500 particles...\n")
t0 <- proc.time()
samples <- monty_sample(posterior, sampler, 5000,
                        initial = matrix(c(0.4, 0.08), 2, 4),
                        n_chains = 4)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("  Done in %.1f seconds\n", elapsed))

# Extract samples: samples$pars is [n_pars, n_steps, n_chains]
beta_samples <- as.vector(samples$pars[1, , ])
gamma_samples <- as.vector(samples$pars[2, , ])

# Discard burn-in (first 1000 per chain)
burnin <- 1000
n_steps <- dim(samples$pars)[2]
n_chains <- dim(samples$pars)[3]
keep <- (burnin + 1):n_steps
beta_post <- as.vector(samples$pars[1, keep, ])
gamma_post <- as.vector(samples$pars[2, keep, ])

cat(sprintf("  Post burn-in samples: %d (per parameter)\n", length(beta_post)))
cat(sprintf("  beta:  mean=%.4f  sd=%.4f  [%.4f, %.4f]\n",
            mean(beta_post), sd(beta_post),
            quantile(beta_post, 0.025), quantile(beta_post, 0.975)))
cat(sprintf("  gamma: mean=%.4f  sd=%.4f  [%.4f, %.4f]\n",
            mean(gamma_post), sd(gamma_post),
            quantile(gamma_post, 0.025), quantile(gamma_post, 0.975)))

# Save posterior samples
write.csv(data.frame(beta = beta_post, gamma = gamma_post),
          "benchmark/posterior_r.csv", row.names = FALSE)
cat("  Saved posterior_r.csv\n")

# Also save log-likelihood distribution for comparison (use different seeds)
cat("  Computing log-likelihood distribution (500 particles × 100 runs)...\n")
ll_vals <- numeric(100)
for (i in seq_len(100)) {
  f_tmp <- dust_filter_create(sir, time_start = 0, data = data_pf,
                              n_particles = 500, seed = NULL)
  ll_vals[i] <- dust_likelihood_run(f_tmp, true_pars)
}
write.csv(data.frame(ll = ll_vals), "benchmark/ll_dist_r.csv", row.names = FALSE)
cat(sprintf("  LL: mean=%.2f  sd=%.2f\n", mean(ll_vals), sd(ll_vals)))
cat("  Saved ll_dist_r.csv\n")
cat("Done.\n")
