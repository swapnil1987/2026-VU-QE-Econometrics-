# ============================================================================
# 02_pooled_ols_vs_fixed_effects.R
# Lecture 1 companion: pooled OLS bias and the logic of fixed effects.
#
# The first block shows one sample.
# The second block runs a Monte Carlo exercise to compare estimators.
# ============================================================================

helper_candidates <- c(
  "panel_helpers.R",
  file.path("lectures", "code", "panel-data", "panel_helpers.R")
)
helper_path <- helper_candidates[file.exists(helper_candidates)][1]
if (is.na(helper_path)) {
  stop("Could not find panel_helpers.R")
}
source(helper_path)

estimate_once <- function(seed_value) {
  panel <- simulate_panel_foundations(
    N = 150,
    T = 6,
    beta = 1.0,
    rho_alpha = 1.40,
    rho_lambda = 0.00,
    serial_u = 0.25,
    seed = seed_value
  )

  pooled_fit <- fit_ols(y ~ x, data = panel)

  panel$y_within <- demean_by(panel$y, panel$id)
  panel$x_within <- demean_by(panel$x, panel$id)
  within_fit <- fit_ols(y_within ~ x_within - 1, data = panel)

  lsdv_fit <- lm(y ~ x + factor(id), data = panel)

  c(
    pooled = pooled_fit$coef[pooled_fit$coef_names == "x"],
    fe_within = within_fit$coef[within_fit$coef_names == "x_within"],
    fe_lsdv = coef(lsdv_fit)[["x"]]
  )
}

single_draw <- estimate_once(202)
single_draw_df <- data.frame(
  estimator = names(single_draw),
  slope = as.numeric(single_draw),
  row.names = NULL
)

panel_message("One simulated sample:")
print(single_draw_df)
panel_message("")
panel_message("Difference between FE via demeaning and FE via unit dummies: ",
              format(abs(single_draw["fe_within"] - single_draw["fe_lsdv"]),
                     digits = 3))

reps <- 300
mc_store <- matrix(NA_real_, nrow = reps, ncol = 3)
colnames(mc_store) <- c("pooled", "fe_within", "fe_lsdv")

for (rep in seq_len(reps)) {
  mc_store[rep, ] <- estimate_once(1000 + rep)
}

mc_summary <- data.frame(
  estimator = colnames(mc_store),
  mean_estimate = colMeans(mc_store),
  bias = colMeans(mc_store) - 1.0,
  sd_estimate = apply(mc_store, 2, sd),
  row.names = NULL
)

panel_message("")
panel_message("Monte Carlo summary across ", reps, " repetitions:")
print(mc_summary)

panel_message("")
panel_message("Interpretation:")
panel_message("- Pooled OLS is too large because x is positively correlated with the unit effect alpha_i.")
panel_message("- FE via demeaning and FE via unit dummies are numerically the same estimator.")
panel_message("- Once identification uses within-unit movements only, the slope centers around the truth.")

boxplot(mc_store,
        col = c("grey75", "lightblue", "lightgreen"),
        main = "Pooled OLS versus fixed effects",
        ylab = "Estimated slope")
abline(h = 1.0, lty = 2, lwd = 2, col = "firebrick")
