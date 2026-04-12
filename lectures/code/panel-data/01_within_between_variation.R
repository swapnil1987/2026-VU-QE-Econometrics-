# ============================================================================
# 01_within_between_variation.R
# Lecture 1 companion: why panel data are useful in the first place.
#
# This script shows that a panel regressor can be split into:
#   1. the grand mean,
#   2. a between-unit component,
#   3. a within-unit component.
# ============================================================================

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("panel_helpers.R")

truth_beta <- 1.0
panel <- simulate_panel_foundations(
  N = 80,
  T = 6,
  beta = truth_beta,
  rho_alpha = 1.25,
  rho_lambda = 0.25,
  serial_u = 0.20,
  seed = 101
)

panel$x_bar <- unit_means(panel$x, panel$id)
panel$y_bar <- unit_means(panel$y, panel$id)
panel$x_within <- panel$x - panel$x_bar
panel$y_within <- panel$y - panel$y_bar

grand_mean_x <- mean(panel$x)
reconstructed_x <- grand_mean_x + (panel$x_bar - grand_mean_x) + panel$x_within
max_decomp_error <- max(abs(panel$x - reconstructed_x))

between_df <- between_panel(panel, y_var = "y", x_var = "x")
pooled_fit <- fit_ols(y ~ x, data = panel)
between_fit <- fit_ols(y_bar ~ x_bar, data = between_df)
within_fit <- fit_ols(y_within ~ x_within - 1, data = panel)

results <- data.frame(
  estimator = c("Pooled OLS", "Between regression", "Within regression"),
  slope = c(
    pooled_fit$coef[pooled_fit$coef_names == "x"],
    between_fit$coef[between_fit$coef_names == "x_bar"],
    within_fit$coef[within_fit$coef_names == "x_within"]
  ),
  row.names = NULL
)

panel_message("Maximum numerical error in the x decomposition: ",
              format(max_decomp_error, digits = 3))
panel_message("")
panel_message("Slope comparison:")
print(results)

panel_message("")
panel_message("Interpretation:")
panel_message("- The between slope is inflated because high-alpha units tend to have high x.")
panel_message("- The within slope is closer to the causal effect because it compares each unit to itself.")
panel_message("- The pooled slope mixes both sources of variation.")

sample_ids <- c(3, 17, 42)
par(mfrow = c(1, length(sample_ids)), mar = c(4, 4, 3, 1))
for (current_id in sample_ids) {
  unit_df <- panel[panel$id == current_id, ]
  plot(unit_df$time, unit_df$x,
       type = "b",
       pch = 19,
       xlab = "Time",
       ylab = "x_it",
       main = paste("Unit", current_id),
       ylim = range(panel$x))
  abline(h = mean(panel$x), lty = 2, col = "grey40")
  abline(h = unique(unit_df$x_bar), lwd = 2, col = "firebrick")
  legend("topleft",
         legend = c("x_it", "Grand mean", "Unit mean"),
         lty = c(1, 2, 1),
         pch = c(19, NA, NA),
         col = c("black", "grey40", "firebrick"),
         bty = "n",
         cex = 0.8)
}
