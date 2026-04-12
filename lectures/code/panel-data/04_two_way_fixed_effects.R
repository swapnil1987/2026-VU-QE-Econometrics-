# ============================================================================
# 04_two_way_fixed_effects.R
# Lecture 1 companion: why time effects matter.
#
# Part A: common time shocks bias one-way FE.
# Part B: variables with only time variation are absorbed by time effects.
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

panel <- simulate_panel_foundations(
  N = 160,
  T = 8,
  beta = 1.0,
  rho_alpha = 1.00,
  rho_lambda = 1.30,
  serial_u = 0.25,
  seed = 404
)

panel$y_oneway <- demean_by(panel$y, panel$id)
panel$x_oneway <- demean_by(panel$x, panel$id)
oneway_fit <- fit_ols(y_oneway ~ x_oneway - 1, data = panel)

panel$y_twoway <- twoway_demean(panel$y, panel$id, panel$time)
panel$x_twoway <- twoway_demean(panel$x, panel$id, panel$time)
twoway_fit <- fit_ols(y_twoway ~ x_twoway - 1, data = panel)

comparison <- data.frame(
  estimator = c("One-way FE", "Two-way FE"),
  slope = c(
    oneway_fit$coef[oneway_fit$coef_names == "x_oneway"],
    twoway_fit$coef[twoway_fit$coef_names == "x_twoway"]
  ),
  row.names = NULL
)

panel_message("Part A: time shocks correlated with x_it")
print(comparison)

panel_message("")
panel_message("Interpretation:")
panel_message("- One-way FE removes alpha_i but still leaves common macro shocks lambda_t in the error.")
panel_message("- Two-way FE removes both alpha_i and lambda_t, so the slope returns closer to the truth.")

time_means_df <- aggregate(cbind(y, x) ~ time, data = panel, FUN = mean)
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
plot(time_means_df$time, time_means_df$y,
     type = "b", pch = 19, xlab = "Time", ylab = "Average y_it",
     main = "Average outcome by period")
plot(time_means_df$time, time_means_df$x,
     type = "b", pch = 19, xlab = "Time", ylab = "Average x_it",
     main = "Average regressor by period")

panel$national_policy <- panel$time
panel$policy_twoway <- twoway_demean(panel$national_policy, panel$id, panel$time)

twfe_lm <- lm(y ~ x + national_policy + factor(id) + factor(time), data = panel)
lm_coef <- coef(twfe_lm)
policy_coef <- if ("national_policy" %in% names(lm_coef)) {
  lm_coef[["national_policy"]]
} else {
  NA_real_
}
part_b <- data.frame(
  statistic = c("Variance of TWFE-transformed national_policy",
                "Coefficient reported by lm"),
  value = c(
    var(panel$policy_twoway),
    policy_coef
  ),
  row.names = NULL
)

panel_message("")
panel_message("Part B: a regressor with only time variation")
print(part_b)
panel_message("- After removing time effects, a purely aggregate regressor has no identifying variation left.")

