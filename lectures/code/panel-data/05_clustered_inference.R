# ============================================================================
# 05_clustered_inference.R
# Lecture 1 companion: why clustered standard errors matter in panel data.
#
# We estimate a fixed-effects model under the null beta = 0 and compare
# rejection rates across three variance estimators:
#   1. homoskedastic iid,
#   2. heteroskedasticity-robust HC1,
#   3. cluster-robust by unit.
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

single_run <- function(seed_value) {
  panel <- simulate_panel_foundations(
    N = 120,
    T = 6,
    beta = 0.0,
    rho_alpha = 0.80,
    rho_lambda = 0.20,
    serial_u = 0.55,
    seed = seed_value
  )

  panel$y_within <- demean_by(panel$y, panel$id)
  panel$x_within <- demean_by(panel$x, panel$id)
  fit <- fit_ols(y_within ~ x_within - 1, data = panel)

  iid_tab <- coef_table(fit, iid_vcov(fit))
  hc1_tab <- coef_table(fit, hc1_vcov(fit))
  clu_tab <- coef_table(fit, cluster_vcov(fit, cluster = panel$id))

  data.frame(
    estimate = fit$coef[fit$coef_names == "x_within"],
    t_iid = extract_term(iid_tab, "x_within")$t_stat,
    t_hc1 = extract_term(hc1_tab, "x_within")$t_stat,
    t_cluster = extract_term(clu_tab, "x_within")$t_stat
  )
}

first_draw <- single_run(505)
panel_message("One fixed-effects regression under the null beta = 0:")
print(first_draw)

reps <- 400
mc <- do.call(
  rbind,
  lapply(seq_len(reps), function(rep) single_run(6000 + rep))
)

rejection_rates <- data.frame(
  variance_estimator = c("IID", "HC1", "Clustered by unit"),
  rejection_rate = c(
    mean(abs(mc$t_iid) > 1.96),
    mean(abs(mc$t_hc1) > 1.96),
    mean(abs(mc$t_cluster) > 1.96)
  ),
  row.names = NULL
)

panel_message("")
panel_message("Rejection rates at the 5 percent level across ", reps, " repetitions:")
print(rejection_rates)

panel_message("")
panel_message("Interpretation:")
panel_message("- The data were generated with beta = 0, so rejection rates should be near 5 percent.")
panel_message("- IID and even HC1 standard errors ignore within-unit dependence after panel transformations.")
panel_message("- Clustered standard errors are better aligned with the repeated-sample uncertainty in a panel.")

barplot(rejection_rates$rejection_rate,
        names.arg = rejection_rates$variance_estimator,
        col = c("grey70", "lightblue", "lightgreen"),
        ylim = c(0, max(rejection_rates$rejection_rate, 0.05) + 0.05),
        ylab = "Rejection rate",
        main = "False rejection under different variance estimators")
abline(h = 0.05, lty = 2, lwd = 2, col = "firebrick")

