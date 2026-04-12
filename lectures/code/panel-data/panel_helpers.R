# ============================================================================
# panel_helpers.R
# Shared utilities for Lecture 1 panel-data companion scripts.
#
# Dependencies: base R only.
# ============================================================================

panel_seed <- function(seed = 123) {
  set.seed(seed)
}

make_balanced_panel <- function(N = 100L, T = 6L) {
  panel <- expand.grid(
    id = seq_len(N),
    time = seq_len(T)
  )
  panel <- panel[order(panel$id, panel$time), ]
  rownames(panel) <- NULL
  panel
}

simulate_panel_foundations <- function(N = 200L,
                                       T = 6L,
                                       beta = 1.0,
                                       rho_alpha = 1.0,
                                       rho_lambda = 1.0,
                                       sigma_alpha = 1.0,
                                       sigma_lambda = 1.0,
                                       sigma_x = 1.0,
                                       sigma_u = 1.0,
                                       serial_u = 0.4,
                                       measurement_sd = 0.0,
                                       seed = 123) {
  panel_seed(seed)
  panel <- make_balanced_panel(N = N, T = T)

  alpha_i <- rnorm(N, mean = 0, sd = sigma_alpha)
  lambda_t <- rnorm(T, mean = 0, sd = sigma_lambda)

  u <- numeric(N * T)
  x_shock <- rnorm(N * T, mean = 0, sd = sigma_x)

  for (i in seq_len(N)) {
    idx <- ((i - 1L) * T + 1L):(i * T)
    u_i <- numeric(T)
    u_i[1] <- rnorm(1, mean = 0, sd = sigma_u)
    if (T >= 2L) {
      for (t in 2:T) {
        innov_t <- rnorm(1, mean = 0, sd = sigma_u)
        u_i[t] <- serial_u * u_i[t - 1L] + innov_t
      }
    }
    u[idx] <- u_i
  }

  alpha <- alpha_i[panel$id]
  lambda <- lambda_t[panel$time]
  x_true <- rho_alpha * alpha + rho_lambda * lambda + x_shock
  x_obs <- x_true + rnorm(N * T, mean = 0, sd = measurement_sd)
  y <- beta * x_true + alpha + lambda + u

  panel$alpha <- alpha
  panel$lambda <- lambda
  panel$u <- u
  panel$x_true <- x_true
  panel$x <- x_obs
  panel$measurement_error <- x_obs - x_true
  panel$y <- y
  panel
}

unit_means <- function(x, id) {
  ave(x, id, FUN = mean)
}

time_means <- function(x, time) {
  ave(x, time, FUN = mean)
}

demean_by <- function(x, group) {
  x - ave(x, group, FUN = mean)
}

twoway_demean <- function(x, id, time) {
  x - unit_means(x, id) - time_means(x, time) + mean(x)
}

first_difference_df <- function(data,
                                vars,
                                id_var = "id",
                                time_var = "time") {
  data <- data[order(data[[id_var]], data[[time_var]]), , drop = FALSE]
  keep <- rep(TRUE, nrow(data))

  for (var in vars) {
    data[[paste0("d_", var)]] <- NA_real_
  }

  id_levels <- unique(data[[id_var]])
  for (id_value in id_levels) {
    idx <- which(data[[id_var]] == id_value)
    keep[idx[1]] <- FALSE
    for (var in vars) {
      data[[paste0("d_", var)]][idx] <- c(NA_real_, diff(data[[var]][idx]))
    }
  }

  rownames(data) <- NULL
  data[keep, , drop = FALSE]
}

fit_ols <- function(formula, data) {
  mf <- model.frame(formula, data = data)
  y <- model.response(mf)
  X <- model.matrix(attr(mf, "terms"), data = mf)
  XtX_inv <- solve(crossprod(X))
  beta_hat <- XtX_inv %*% crossprod(X, y)
  fitted <- as.vector(X %*% beta_hat)
  resid <- as.vector(y - fitted)

  list(
    formula = formula,
    X = X,
    y = y,
    fitted = fitted,
    resid = resid,
    coef = as.vector(beta_hat),
    coef_names = colnames(X),
    n = nrow(X),
    k = ncol(X)
  )
}

iid_vcov <- function(fit) {
  sigma2_hat <- sum(fit$resid^2) / (fit$n - fit$k)
  sigma2_hat * solve(crossprod(fit$X))
}

hc1_vcov <- function(fit) {
  XtX_inv <- solve(crossprod(fit$X))
  meat <- crossprod(fit$X * fit$resid)
  adjustment <- fit$n / (fit$n - fit$k)
  XtX_inv %*% (adjustment * meat) %*% XtX_inv
}

cluster_vcov <- function(fit, cluster) {
  cluster <- as.factor(cluster)
  XtX_inv <- solve(crossprod(fit$X))
  meat <- matrix(0, nrow = fit$k, ncol = fit$k)

  for (g in levels(cluster)) {
    idx <- which(cluster == g)
    Xg <- fit$X[idx, , drop = FALSE]
    ug <- matrix(fit$resid[idx], ncol = 1)
    meat <- meat + crossprod(Xg, ug %*% t(ug) %*% Xg)
  }

  G <- nlevels(cluster)
  adjustment <- (G / (G - 1)) * ((fit$n - 1) / (fit$n - fit$k))
  XtX_inv %*% (adjustment * meat) %*% XtX_inv
}

coef_table <- function(fit, vcov_mat) {
  se <- sqrt(diag(vcov_mat))
  t_stat <- fit$coef / se
  data.frame(
    term = fit$coef_names,
    estimate = fit$coef,
    std_error = se,
    t_stat = t_stat,
    row.names = NULL
  )
}

extract_term <- function(table_df, term_name) {
  table_df[table_df$term == term_name, , drop = FALSE]
}

between_panel <- function(data, y_var = "y", x_var = "x", id_var = "id") {
  data.frame(
    id = unique(data[[id_var]]),
    y_bar = tapply(data[[y_var]], data[[id_var]], mean),
    x_bar = tapply(data[[x_var]], data[[id_var]], mean),
    row.names = NULL
  )
}

panel_message <- function(...) {
  cat(paste0(..., "\n"))
}
