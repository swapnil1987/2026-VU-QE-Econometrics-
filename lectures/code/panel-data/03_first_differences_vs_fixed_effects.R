# ============================================================================
# 03_first_differences_vs_fixed_effects.R
# Lecture 1 companion: FE versus FD.
#
# Part A: when T = 2, FE and FD are exactly the same.
# Part B: with noisy regressors, FD can be more attenuated than FE.
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

panel_t2 <- simulate_panel_foundations(
  N = 250,
  T = 2,
  beta = 1.2,
  rho_alpha = 1.10,
  rho_lambda = 0.00,
  serial_u = 0.20,
  seed = 303
)

panel_t2$y_within <- demean_by(panel_t2$y, panel_t2$id)
panel_t2$x_within <- demean_by(panel_t2$x, panel_t2$id)
fe_t2 <- fit_ols(y_within ~ x_within - 1, data = panel_t2)

fd_t2 <- first_difference_df(panel_t2, vars = c("y", "x"))
fd_fit_t2 <- fit_ols(d_y ~ d_x - 1, data = fd_t2)

part_a <- data.frame(
  estimator = c("Fixed effects", "First differences"),
  slope = c(
    fe_t2$coef[fe_t2$coef_names == "x_within"],
    fd_fit_t2$coef[fd_fit_t2$coef_names == "d_x"]
  ),
  row.names = NULL
)

panel_message("Part A: T = 2")
print(part_a)
panel_message("Absolute difference: ",
              format(abs(part_a$slope[1] - part_a$slope[2]), digits = 3))

estimate_noisy_once <- function(seed_value) {
  panel <- simulate_panel_foundations(
    N = 180,
    T = 6,
    beta = 1.0,
    rho_alpha = 1.00,
    rho_lambda = 0.00,
    serial_u = 0.20,
    measurement_sd = 1.25,
    seed = seed_value
  )

  panel$y_within <- demean_by(panel$y, panel$id)
  panel$x_within <- demean_by(panel$x, panel$id)
  fe_fit <- fit_ols(y_within ~ x_within - 1, data = panel)

  fd_panel <- first_difference_df(panel, vars = c("y", "x"))
  fd_fit <- fit_ols(d_y ~ d_x - 1, data = fd_panel)

  c(
    fe = fe_fit$coef[fe_fit$coef_names == "x_within"],
    fd = fd_fit$coef[fd_fit$coef_names == "d_x"]
  )
}

reps <- 300
mc_noisy <- matrix(NA_real_, nrow = reps, ncol = 2)
colnames(mc_noisy) <- c("fe", "fd")

for (rep in seq_len(reps)) {
  mc_noisy[rep, ] <- estimate_noisy_once(4000 + rep)
}

part_b <- data.frame(
  estimator = colnames(mc_noisy),
  mean_estimate = colMeans(mc_noisy),
  bias = colMeans(mc_noisy) - 1.0,
  sd_estimate = apply(mc_noisy, 2, sd),
  row.names = NULL
)

panel_message("")
panel_message("Part B: noisy x_it")
print(part_b)

panel_message("")
panel_message("Interpretation:")
panel_message("- When T = 2, FE and FD are just two ways of extracting the same within-unit movement.")
panel_message("- With more periods, FE and FD weight time differently.")
panel_message("- Once x_it is noisy, differencing doubles the transitory noise and FD is typically pulled down more.")

boxplot(mc_noisy,
        col = c("lightblue", "lightgreen"),
        main = "FE versus FD with measurement error",
        ylab = "Estimated slope")
abline(h = 1.0, lty = 2, lwd = 2, col = "firebrick")

