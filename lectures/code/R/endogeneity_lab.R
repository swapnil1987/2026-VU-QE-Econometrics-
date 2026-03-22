# ============================================================================
# endogeneity_lab.R
# R translation of endogeneity_lab.py
#
# Provides simulation, estimation, and plotting utilities for teaching
# endogeneity in econometrics.
#
# Dependencies: ggplot2, patchwork
# ============================================================================

library(ggplot2)
library(patchwork)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

set_plot_style <- function() {
  theme_set(
    theme_minimal(base_size = 12) +
      theme(
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(colour = "grey90"),
        axis.line        = element_line(colour = "black", linewidth = 0.4),
        legend.background = element_blank(),
        plot.title       = element_text(face = "bold", size = 13)
      )
  )
}

# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

slope_with_intercept <- function(x, y) {
  cov(x, y) / var(x)
}

multiple_ols_coef <- function(y, X_mat, coef_index = 1L) {
  # X_mat is a matrix of regressors (without intercept)
  df <- data.frame(y = y, X_mat)
  fit <- lm(y ~ ., data = df)
  unname(coef(fit)[coef_index + 1L])
}

iv_ratio <- function(y, x, z) {
  denom <- cov(z, x)
  if (abs(denom) < 1e-12) return(NA_real_)
  cov(z, y) / denom
}

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

summarize_estimates <- function(estimates, truth) {
  # estimates: named list of numeric vectors
  rows <- lapply(names(estimates), function(nm) {
    arr <- estimates[[nm]]
    arr <- arr[is.finite(arr)]
    data.frame(
      Estimator  = nm,
      Mean       = mean(arr),
      Bias       = mean(arr) - truth,
      Std.dev    = sd(arr),
      RMSE       = sqrt(mean((arr - truth)^2)),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

# ---------------------------------------------------------------------------
# Distribution plot
# ---------------------------------------------------------------------------

plot_estimate_distributions <- function(estimates, truth, title,
                                        x_label = "Estimate",
                                        bins = 40, xlim_range = NULL) {
  colors <- c("#c44e52", "#4c72b0", "#55a868", "#8172b3", "#ccb974")

  # Build long data frame
  dfs <- mapply(function(nm, i) {
    arr <- estimates[[nm]]
    arr <- arr[is.finite(arr)]
    data.frame(
      estimator = paste0(nm, ": mean = ", sprintf("%.3f", mean(arr))),
      value     = arr,
      stringsAsFactors = FALSE
    )
  }, names(estimates), seq_along(estimates), SIMPLIFY = FALSE)

  plot_df <- do.call(rbind, dfs)
  plot_df$estimator <- factor(plot_df$estimator, levels = unique(plot_df$estimator))

  color_map <- setNames(colors[seq_along(estimates)], levels(plot_df$estimator))

  p <- ggplot(plot_df, aes(x = value, fill = estimator)) +
    geom_histogram(aes(y = after_stat(density)),
                   bins = bins, alpha = 0.35, position = "identity") +
    geom_vline(xintercept = truth, linetype = "dashed", linewidth = 1,
               colour = "black") +
    scale_fill_manual(values = color_map) +
    labs(title = title, x = x_label, y = "Density", fill = NULL) +
    theme(legend.position = "bottom")

  if (!is.null(xlim_range)) {
    p <- p + coord_cartesian(xlim = xlim_range)
  }

  print(p)
  invisible(p)
}

# ============================================================================
# 1. BASIC ENDOGENEITY
# ============================================================================

sample_basic_endogeneity <- function(n = 400, beta = 1.0, gamma = 0.0,
                                     signal_strength = 1.0, noise_sd = 1.0,
                                     seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  z <- rnorm(n)
  u <- rnorm(n)
  v <- rnorm(n, sd = noise_sd)
  x <- signal_strength * z + gamma * u + v
  y <- beta * x + u
  data.frame(z = z, u = u, x = x, y = y)
}

basic_case_summary <- function(data, beta) {
  ols_beta <- slope_with_intercept(data$x, data$y)
  corr_x_u <- cor(data$x, data$u)
  data.frame(
    Statistic = c("True beta", "OLS slope", "Bias", "corr(X, u)"),
    Value     = c(beta, ols_beta, ols_beta - beta, corr_x_u)
  )
}

plot_basic_case <- function(data, beta, title) {
  fit <- lm(y ~ x, data = data)
  x_grid <- seq(min(data$x), max(data$x), length.out = 200)
  fitted_line <- coef(fit)[1] + coef(fit)[2] * x_grid
  true_line   <- beta * x_grid

  line_df <- data.frame(
    x = rep(x_grid, 2),
    y = c(fitted_line, true_line),
    type = rep(c("OLS fit", "True line"), each = length(x_grid))
  )

  p1 <- ggplot(data, aes(x = x, y = y)) +
    geom_point(alpha = 0.35, size = 1, colour = "#4c72b0") +
    geom_line(data = line_df, aes(x = x, y = y, colour = type, linetype = type),
              linewidth = 1) +
    scale_colour_manual(values = c("OLS fit" = "#c44e52", "True line" = "black")) +
    scale_linetype_manual(values = c("OLS fit" = "solid", "True line" = "dashed")) +
    labs(title = paste0(title, ": Y against X"), x = "X", y = "Y",
         colour = NULL, linetype = NULL) +
    theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))

  corr_label <- sprintf("corr(X, u) = %.3f", cor(data$x, data$u))
  p2 <- ggplot(data, aes(x = x, y = u)) +
    geom_point(alpha = 0.35, size = 1, colour = "#55a868") +
    geom_hline(yintercept = 0, linewidth = 0.5) +
    geom_vline(xintercept = 0, linewidth = 0.5) +
    annotate("label", x = min(data$x) + 0.05 * diff(range(data$x)),
             y = max(data$u) * 0.95,
             label = corr_label, hjust = 0, vjust = 1,
             fill = "white", label.size = 0) +
    labs(title = paste0(title, ": hidden error u against X"), x = "X", y = "u")

  print(p1 + p2)
  invisible(list(p1 = p1, p2 = p2))
}

mc_basic_endogeneity <- function(n = 400, beta = 1.0,
                                 gamma_values = c(-0.9, -0.5, 0.0, 0.5, 0.9),
                                 reps = 600, signal_strength = 1.0,
                                 noise_sd = 1.0, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  rows <- lapply(gamma_values, function(gam) {
    slopes <- numeric(reps)
    corrs  <- numeric(reps)
    for (rep in seq_len(reps)) {
      z <- rnorm(n)
      u <- rnorm(n)
      v <- rnorm(n, sd = noise_sd)
      x <- signal_strength * z + gam * u + v
      y <- beta * x + u
      slopes[rep] <- slope_with_intercept(x, y)
      corrs[rep]  <- cor(x, u)
    }
    data.frame(
      gamma              = gam,
      Average.corr.X.u   = mean(corrs),
      Mean.OLS.slope      = mean(slopes),
      Bias               = mean(slopes) - beta,
      Std.dev.of.OLS     = sd(slopes)
    )
  })

  do.call(rbind, rows)
}

plot_basic_monte_carlo <- function(results, beta) {
  p1 <- ggplot(results, aes(x = gamma, y = Mean.OLS.slope)) +
    geom_line(colour = "#4c72b0", linewidth = 1) +
    geom_point(colour = "#4c72b0", size = 3) +
    geom_hline(yintercept = beta, linetype = "dashed", linewidth = 1) +
    labs(title = "Average OLS estimate",
         x = expression("Strength of endogeneity " * gamma),
         y = "Mean OLS slope")

  p2 <- ggplot(results, aes(x = gamma, y = Average.corr.X.u)) +
    geom_line(colour = "#c44e52", linewidth = 1) +
    geom_point(colour = "#c44e52", size = 3) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = 1) +
    labs(title = "Average correlation between X and u",
         x = expression("Strength of endogeneity " * gamma),
         y = "corr(X, u)")

  print(p1 + p2)
  invisible(list(p1 = p1, p2 = p2))
}

# ============================================================================
# 2. OMITTED VARIABLE BIAS
# ============================================================================

sample_ovb <- function(n = 400, beta_x = 1.0, beta_w = 1.5, rho_xw = 0.7,
                       instrument_strength = 0.9, x_noise_sd = 1.0,
                       seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  z   <- rnorm(n)
  w   <- rnorm(n)
  v   <- rnorm(n, sd = x_noise_sd)
  eps <- rnorm(n)
  x   <- instrument_strength * z + rho_xw * w + v
  y   <- beta_x * x + beta_w * w + eps
  data.frame(z = z, w = w, x = x, y = y)
}

ovb_one_run <- function(n = 400, beta_x = 1.0, beta_w = 1.5, rho_xw = 0.7,
                        instrument_strength = 0.9, x_noise_sd = 1.0,
                        seed = NULL) {
  data <- sample_ovb(n = n, beta_x = beta_x, beta_w = beta_w, rho_xw = rho_xw,
                     instrument_strength = instrument_strength,
                     x_noise_sd = x_noise_sd, seed = seed)

  x <- data$x;  y <- data$y;  w <- data$w;  z <- data$z
  var_x <- instrument_strength^2 + rho_xw^2 + x_noise_sd^2
  theory_omit_w <- beta_x + beta_w * rho_xw / var_x

  summary_df <- data.frame(
    Estimator = c("Truth", "Theory: OLS omit W", "OLS omit W",
                  "OLS control for W", "IV using Z", "corr(X, W)"),
    Estimate  = c(beta_x,
                  theory_omit_w,
                  slope_with_intercept(x, y),
                  multiple_ols_coef(y, cbind(x, w), coef_index = 1L),
                  iv_ratio(y, x, z),
                  cor(x, w))
  )
  list(data = data, summary = summary_df)
}

plot_ovb_sample <- function(data, beta_x) {
  x <- data$x;  w <- data$w;  y <- data$y

  corr_label <- sprintf("corr(X, W) = %.3f", cor(x, w))
  p1 <- ggplot(data, aes(x = w, y = x)) +
    geom_point(alpha = 0.4, size = 1, colour = "#55a868") +
    annotate("label", x = min(w) + 0.05 * diff(range(w)),
             y = max(x) * 0.95,
             label = corr_label, hjust = 0, vjust = 1,
             fill = "white", label.size = 0) +
    labs(title = "The omitted variable W is correlated with X",
         x = "W (unobserved in the short regression)", y = "X")

  short_fit <- lm(y ~ x, data = data)
  long_fit  <- lm(y ~ x + w, data = data)
  x_grid    <- seq(min(x), max(x), length.out = 200)

  line_df <- data.frame(
    xv = rep(x_grid, 3),
    yv = c(coef(short_fit)[1] + coef(short_fit)[2] * x_grid,
           coef(long_fit)[1]  + coef(long_fit)[2]  * x_grid,
           beta_x * x_grid),
    type = rep(c("OLS omit W", "OLS control W", "True slope"), each = length(x_grid))
  )
  line_df$type <- factor(line_df$type,
                         levels = c("OLS omit W", "OLS control W", "True slope"))

  p2 <- ggplot(data, aes(x = x, y = y)) +
    geom_point(alpha = 0.35, size = 1, colour = "#4c72b0") +
    geom_line(data = line_df,
              aes(x = xv, y = yv, colour = type, linetype = type),
              linewidth = 1) +
    scale_colour_manual(values = c("OLS omit W" = "#c44e52",
                                   "OLS control W" = "#55a868",
                                   "True slope" = "black")) +
    scale_linetype_manual(values = c("OLS omit W" = "solid",
                                     "OLS control W" = "solid",
                                     "True slope" = "dashed")) +
    labs(title = "Short regression versus the truth",
         x = "X", y = "Y", colour = NULL, linetype = NULL) +
    theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1))

  print(p1 + p2)
  invisible(list(p1 = p1, p2 = p2))
}

mc_ovb <- function(n = 400, beta_x = 1.0, beta_w = 1.5, rho_xw = 0.7,
                   instrument_strength = 0.9, x_noise_sd = 1.0,
                   reps = 800, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  ols_short <- numeric(reps)
  ols_long  <- numeric(reps)
  iv_est    <- numeric(reps)

  for (rep in seq_len(reps)) {
    z   <- rnorm(n)
    w   <- rnorm(n)
    v   <- rnorm(n, sd = x_noise_sd)
    eps <- rnorm(n)
    x   <- instrument_strength * z + rho_xw * w + v
    y   <- beta_x * x + beta_w * w + eps
    ols_short[rep] <- slope_with_intercept(x, y)
    ols_long[rep]  <- multiple_ols_coef(y, cbind(x, w), coef_index = 1L)
    iv_est[rep]    <- iv_ratio(y, x, z)
  }

  list("OLS omit W"    = ols_short,
       "OLS control W" = ols_long,
       "IV using Z"    = iv_est)
}

# ============================================================================
# 3. MEASUREMENT ERROR
# ============================================================================

sample_measurement_error <- function(n = 400, beta_x = 1.0,
                                     instrument_strength = 0.9,
                                     latent_noise_sd = 1.0,
                                     measurement_noise_sd = 1.0,
                                     seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  z               <- rnorm(n)
  latent_noise    <- rnorm(n, sd = latent_noise_sd)
  x_true          <- instrument_strength * z + latent_noise
  measurement_err <- rnorm(n, sd = measurement_noise_sd)
  eps             <- rnorm(n)
  x_observed      <- x_true + measurement_err
  y               <- beta_x * x_true + eps

  data.frame(z = z, x_true = x_true, x_observed = x_observed,
             measurement_error = measurement_err, y = y)
}

measurement_error_one_run <- function(n = 400, beta_x = 1.0,
                                      instrument_strength = 0.9,
                                      latent_noise_sd = 1.0,
                                      measurement_noise_sd = 1.0,
                                      seed = NULL) {
  data <- sample_measurement_error(
    n = n, beta_x = beta_x, instrument_strength = instrument_strength,
    latent_noise_sd = latent_noise_sd, measurement_noise_sd = measurement_noise_sd,
    seed = seed
  )

  x_true     <- data$x_true
  x_observed <- data$x_observed
  y          <- data$y
  z          <- data$z
  theory     <- beta_x * var(x_true) / (var(x_true) + measurement_noise_sd^2)

  summary_df <- data.frame(
    Estimator = c("Truth", "Theory: OLS with noisy X", "OLS using noisy X",
                  "OLS using true X*", "IV using Z",
                  "corr(X observed, measurement error)"),
    Estimate  = c(beta_x,
                  theory,
                  slope_with_intercept(x_observed, y),
                  slope_with_intercept(x_true, y),
                  iv_ratio(y, x_observed, z),
                  cor(x_observed, data$measurement_error))
  )
  list(data = data, summary = summary_df)
}

plot_measurement_error_sample <- function(data, beta_x) {
  x_true     <- data$x_true
  x_observed <- data$x_observed
  y          <- data$y
  x_min <- min(c(x_true, x_observed))
  x_max <- max(c(x_true, x_observed))
  x_grid <- seq(x_min, x_max, length.out = 200)

  p1 <- ggplot(data, aes(x = x_true, y = x_observed)) +
    geom_point(alpha = 0.4, size = 1, colour = "#4c72b0") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 1) +
    labs(title = "Observed X is a noisy measure of true X*",
         x = "True X*", y = "Observed X")

  noisy_fit <- lm(y ~ x_observed, data = data)
  true_fit  <- lm(y ~ x_true, data = data)

  scatter_df <- data.frame(
    xv   = c(x_observed, x_true),
    y    = rep(y, 2),
    type = rep(c("Observed X", "True X*"), each = length(y))
  )

  line_df <- data.frame(
    xv   = rep(x_grid, 3),
    yv   = c(coef(noisy_fit)[1] + coef(noisy_fit)[2] * x_grid,
             coef(true_fit)[1]  + coef(true_fit)[2]  * x_grid,
             beta_x * x_grid),
    type = rep(c("OLS using noisy X", "OLS using true X*", "True slope"),
               each = length(x_grid))
  )
  line_df$type <- factor(line_df$type,
                         levels = c("OLS using noisy X", "OLS using true X*",
                                    "True slope"))

  p2 <- ggplot() +
    geom_point(data = data, aes(x = x_observed, y = y),
               alpha = 0.25, size = 1, colour = "#c44e52") +
    geom_point(data = data, aes(x = x_true, y = y),
               alpha = 0.25, size = 1, colour = "#55a868") +
    geom_line(data = line_df,
              aes(x = xv, y = yv, colour = type, linetype = type),
              linewidth = 1) +
    scale_colour_manual(values = c("OLS using noisy X" = "#c44e52",
                                   "OLS using true X*" = "#55a868",
                                   "True slope" = "black")) +
    scale_linetype_manual(values = c("OLS using noisy X" = "solid",
                                     "OLS using true X*" = "solid",
                                     "True slope" = "dashed")) +
    labs(title = "Measurement error flattens the fitted line",
         x = "Regressor value", y = "Y", colour = NULL, linetype = NULL) +
    theme(legend.position = "bottom")

  print(p1 + p2)
  invisible(list(p1 = p1, p2 = p2))
}

mc_measurement_error <- function(n = 400, beta_x = 1.0,
                                 instrument_strength = 0.9,
                                 latent_noise_sd = 1.0,
                                 measurement_noise_sd = 1.0,
                                 reps = 800, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  ols_noisy <- numeric(reps)
  ols_true  <- numeric(reps)
  iv_est    <- numeric(reps)

  for (rep in seq_len(reps)) {
    z               <- rnorm(n)
    latent_noise    <- rnorm(n, sd = latent_noise_sd)
    x_true          <- instrument_strength * z + latent_noise
    measurement_err <- rnorm(n, sd = measurement_noise_sd)
    eps             <- rnorm(n)
    x_observed      <- x_true + measurement_err
    y               <- beta_x * x_true + eps
    ols_noisy[rep]  <- slope_with_intercept(x_observed, y)
    ols_true[rep]   <- slope_with_intercept(x_true, y)
    iv_est[rep]     <- iv_ratio(y, x_observed, z)
  }

  list("OLS using noisy X" = ols_noisy,
       "OLS using true X*" = ols_true,
       "IV using Z"        = iv_est)
}

# ============================================================================
# 4. SIMULTANEITY
# ============================================================================

sample_simultaneity <- function(n = 400, demand_intercept = 10.0,
                                demand_slope = -0.8, supply_intercept = 2.0,
                                supply_slope = 1.2, instrument_strength = 1.5,
                                demand_shock_sd = 1.0, supply_shock_sd = 1.0,
                                seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  z   <- rnorm(n)
  u_d <- rnorm(n, sd = demand_shock_sd)
  u_s <- rnorm(n, sd = supply_shock_sd)

  price <- (supply_intercept - demand_intercept +
              instrument_strength * z + u_s - u_d) /
    (demand_slope - supply_slope)
  quantity <- demand_intercept + demand_slope * price + u_d

  data.frame(z = z, price = price, quantity = quantity, u_d = u_d, u_s = u_s)
}

simultaneity_one_run <- function(n = 400, demand_intercept = 10.0,
                                 demand_slope = -0.8, supply_intercept = 2.0,
                                 supply_slope = 1.2, instrument_strength = 1.5,
                                 demand_shock_sd = 1.0, supply_shock_sd = 1.0,
                                 seed = NULL) {
  data <- sample_simultaneity(
    n = n, demand_intercept = demand_intercept, demand_slope = demand_slope,
    supply_intercept = supply_intercept, supply_slope = supply_slope,
    instrument_strength = instrument_strength,
    demand_shock_sd = demand_shock_sd, supply_shock_sd = supply_shock_sd,
    seed = seed
  )

  price    <- data$price
  quantity <- data$quantity
  z        <- data$z
  u_d      <- data$u_d

  summary_df <- data.frame(
    Estimator = c("Truth: demand slope", "OLS of Q on P",
                  "IV using supply shifter Z",
                  "corr(P, demand shock)", "corr(Z, P)"),
    Estimate  = c(demand_slope,
                  slope_with_intercept(price, quantity),
                  iv_ratio(quantity, price, z),
                  cor(price, u_d),
                  cor(z, price))
  )
  list(data = data, summary = summary_df)
}

plot_supply_demand_worlds <- function(demand_intercept = 10.0,
                                     demand_slope = -0.8,
                                     supply_intercept = 2.0,
                                     supply_slope = 1.2,
                                     n_points = 30, shock_sd = 2.0,
                                     seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  q_grid <- seq(2, 14, length.out = 200)

  price_from_demand <- function(q, shift = 0) {
    (q - demand_intercept - shift) / demand_slope
  }
  price_from_supply <- function(q, shift = 0) {
    (q - supply_intercept - shift) / supply_slope
  }

  demand_shocks <- rnorm(n_points, sd = shock_sd)
  supply_shocks <- rnorm(n_points, sd = shock_sd)

  # Curves data
  curves_df <- data.frame(
    q = rep(q_grid, 2),
    p = c(price_from_demand(q_grid), price_from_supply(q_grid)),
    curve = rep(c("Demand", "Supply"), each = length(q_grid))
  )

  # Panel 1: Both curves shift
  price_both    <- (supply_intercept - demand_intercept + supply_shocks - demand_shocks) /
    (demand_slope - supply_slope)
  quantity_both <- demand_intercept + demand_slope * price_both + demand_shocks

  p1 <- ggplot() +
    geom_point(data = data.frame(q = quantity_both, p = price_both),
               aes(x = q, y = p), colour = "#c44e52", alpha = 0.7, size = 2) +
    geom_line(data = curves_df, aes(x = q, y = p, colour = curve), linewidth = 1) +
    scale_colour_manual(values = c("Demand" = "#4c72b0", "Supply" = "#55a868")) +
    labs(title = "Both curves shift", x = "Quantity", y = "Price", colour = NULL) +
    theme(legend.position = "bottom")

  # Panel 2: Only supply shifts
  supply_only    <- rnorm(n_points, sd = shock_sd)
  price_supply   <- (supply_intercept - demand_intercept + supply_only) /
    (demand_slope - supply_slope)
  quantity_supply <- demand_intercept + demand_slope * price_supply

  p2 <- ggplot() +
    geom_point(data = data.frame(q = quantity_supply, p = price_supply),
               aes(x = q, y = p), colour = "#4c72b0", alpha = 0.7, size = 2) +
    geom_line(data = curves_df, aes(x = q, y = p, colour = curve), linewidth = 1) +
    scale_colour_manual(values = c("Demand" = "#4c72b0", "Supply" = "#55a868")) +
    labs(title = "Only supply shifts", x = "Quantity", y = "Price", colour = NULL) +
    theme(legend.position = "bottom")

  # Panel 3: Only demand shifts
  demand_only    <- rnorm(n_points, sd = shock_sd)
  price_demand   <- (supply_intercept - demand_intercept - demand_only) /
    (demand_slope - supply_slope)
  quantity_demand <- supply_intercept + supply_slope * price_demand

  p3 <- ggplot() +
    geom_point(data = data.frame(q = quantity_demand, p = price_demand),
               aes(x = q, y = p), colour = "#55a868", alpha = 0.7, size = 2) +
    geom_line(data = curves_df, aes(x = q, y = p, colour = curve), linewidth = 1) +
    scale_colour_manual(values = c("Demand" = "#4c72b0", "Supply" = "#55a868")) +
    labs(title = "Only demand shifts", x = "Quantity", y = "Price", colour = NULL) +
    theme(legend.position = "bottom")

  combined <- p1 + p2 + p3 +
    plot_annotation(
      title = "Why simultaneity is different from omitted-variable bias"
    ) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")

  print(combined)
  invisible(list(p1 = p1, p2 = p2, p3 = p3))
}

mc_simultaneity <- function(n = 400, demand_intercept = 10.0,
                            demand_slope = -0.8, supply_intercept = 2.0,
                            supply_slope = 1.2, instrument_strength = 1.5,
                            demand_shock_sd = 1.0, supply_shock_sd = 1.0,
                            reps = 800, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  ols_est <- numeric(reps)
  iv_est  <- numeric(reps)

  for (rep in seq_len(reps)) {
    z   <- rnorm(n)
    u_d <- rnorm(n, sd = demand_shock_sd)
    u_s <- rnorm(n, sd = supply_shock_sd)
    price <- (supply_intercept - demand_intercept +
                instrument_strength * z + u_s - u_d) /
      (demand_slope - supply_slope)
    quantity <- demand_intercept + demand_slope * price + u_d
    ols_est[rep] <- slope_with_intercept(price, quantity)
    iv_est[rep]  <- iv_ratio(quantity, price, z)
  }

  list("OLS of Q on P"             = ols_est,
       "IV using supply shifter Z" = iv_est)
}
