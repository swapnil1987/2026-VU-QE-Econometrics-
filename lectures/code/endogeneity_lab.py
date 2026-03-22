from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 120,
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _slope_with_intercept(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))


def _multiple_ols_coef(y: np.ndarray, regressors: np.ndarray, coef_index: int = 0) -> float:
    result = sm.OLS(y, sm.add_constant(regressors)).fit()
    return float(result.params[coef_index + 1])


def _iv_ratio(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    denom = np.cov(z, x, ddof=1)[0, 1]
    if np.isclose(denom, 0.0):
        return np.nan
    return float(np.cov(z, y, ddof=1)[0, 1] / denom)


def summarize_estimates(estimates: Mapping[str, np.ndarray], truth: float) -> pd.DataFrame:
    rows = []
    for name, values in estimates.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        rows.append(
            {
                "Estimator": name,
                "Mean": arr.mean(),
                "Bias": arr.mean() - truth,
                "Std. dev.": arr.std(ddof=1),
                "RMSE": np.sqrt(np.mean((arr - truth) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def plot_estimate_distributions(
    estimates: Mapping[str, np.ndarray],
    truth: float,
    title: str,
    x_label: str = "Estimate",
    bins: int = 40,
    xlim: tuple[float, float] | None = None,
) -> None:
    colors = ["#c44e52", "#4c72b0", "#55a868", "#8172b3", "#ccb974"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for color, (name, values) in zip(colors, estimates.items()):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
            label=f"{name}: mean = {arr.mean():.3f}",
        )

    ax.axvline(truth, color="black", linestyle="--", linewidth=2, label=f"Truth = {truth:.3f}")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def sample_basic_endogeneity(
    n: int = 400,
    beta: float = 1.0,
    gamma: float = 0.0,
    signal_strength: float = 1.0,
    noise_sd: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = _rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    v = rng.normal(scale=noise_sd, size=n)
    x = signal_strength * z + gamma * u + v
    y = beta * x + u
    return pd.DataFrame({"z": z, "u": u, "x": x, "y": y})


def basic_case_summary(data: pd.DataFrame, beta: float) -> pd.DataFrame:
    ols_beta = _slope_with_intercept(data["x"].to_numpy(), data["y"].to_numpy())
    corr_x_u = np.corrcoef(data["x"], data["u"])[0, 1]
    return pd.DataFrame(
        [
            {"Statistic": "True beta", "Value": beta},
            {"Statistic": "OLS slope", "Value": ols_beta},
            {"Statistic": "Bias", "Value": ols_beta - beta},
            {"Statistic": "corr(X, u)", "Value": corr_x_u},
        ]
    )


def plot_basic_case(data: pd.DataFrame, beta: float, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    u = data["u"].to_numpy()
    ols_result = sm.OLS(y, sm.add_constant(x)).fit()
    x_grid = np.linspace(x.min(), x.max(), 200)

    axes[0].scatter(x, y, alpha=0.35, s=18, color="#4c72b0")
    axes[0].plot(x_grid, ols_result.params[0] + ols_result.params[1] * x_grid, color="#c44e52", linewidth=2, label="OLS fit")
    axes[0].plot(x_grid, beta * x_grid, color="black", linestyle="--", linewidth=2, label="True line")
    axes[0].set_title(f"{title}: Y against X")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].legend(frameon=False)

    axes[1].scatter(x, u, alpha=0.35, s=18, color="#55a868")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title(f"{title}: hidden error u against X")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("u")
    axes[1].text(
        0.03,
        0.95,
        f"corr(X, u) = {np.corrcoef(x, u)[0, 1]:.3f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    plt.tight_layout()
    plt.show()


def mc_basic_endogeneity(
    n: int = 400,
    beta: float = 1.0,
    gamma_values: tuple[float, ...] = (-0.9, -0.5, 0.0, 0.5, 0.9),
    reps: int = 600,
    signal_strength: float = 1.0,
    noise_sd: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = _rng(seed)
    rows = []

    for gamma in gamma_values:
        slopes = np.empty(reps)
        corrs = np.empty(reps)
        for rep in range(reps):
            z = rng.normal(size=n)
            u = rng.normal(size=n)
            v = rng.normal(scale=noise_sd, size=n)
            x = signal_strength * z + gamma * u + v
            y = beta * x + u
            slopes[rep] = _slope_with_intercept(x, y)
            corrs[rep] = np.corrcoef(x, u)[0, 1]

        rows.append(
            {
                "gamma": gamma,
                "Average corr(X,u)": corrs.mean(),
                "Mean OLS slope": slopes.mean(),
                "Bias": slopes.mean() - beta,
                "Std. dev. of OLS": slopes.std(ddof=1),
            }
        )

    return pd.DataFrame(rows)


def plot_basic_monte_carlo(results: pd.DataFrame, beta: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(results["gamma"], results["Mean OLS slope"], marker="o", linewidth=2, color="#4c72b0")
    axes[0].axhline(beta, color="black", linestyle="--", linewidth=2)
    axes[0].set_title("Average OLS estimate")
    axes[0].set_xlabel("Strength of endogeneity gamma")
    axes[0].set_ylabel("Mean OLS slope")

    axes[1].plot(results["gamma"], results["Average corr(X,u)"], marker="o", linewidth=2, color="#c44e52")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=2)
    axes[1].set_title("Average correlation between X and u")
    axes[1].set_xlabel("Strength of endogeneity gamma")
    axes[1].set_ylabel("corr(X, u)")

    plt.tight_layout()
    plt.show()


def sample_ovb(
    n: int = 400,
    beta_x: float = 1.0,
    beta_w: float = 1.5,
    rho_xw: float = 0.7,
    instrument_strength: float = 0.9,
    x_noise_sd: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = _rng(seed)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    v = rng.normal(scale=x_noise_sd, size=n)
    eps = rng.normal(size=n)
    x = instrument_strength * z + rho_xw * w + v
    y = beta_x * x + beta_w * w + eps
    return pd.DataFrame({"z": z, "w": w, "x": x, "y": y})


def ovb_one_run(
    n: int = 400,
    beta_x: float = 1.0,
    beta_w: float = 1.5,
    rho_xw: float = 0.7,
    instrument_strength: float = 0.9,
    x_noise_sd: float = 1.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = sample_ovb(
        n=n,
        beta_x=beta_x,
        beta_w=beta_w,
        rho_xw=rho_xw,
        instrument_strength=instrument_strength,
        x_noise_sd=x_noise_sd,
        seed=seed,
    )

    x = data["x"].to_numpy()
    y = data["y"].to_numpy()
    w = data["w"].to_numpy()
    z = data["z"].to_numpy()
    var_x = instrument_strength**2 + rho_xw**2 + x_noise_sd**2
    theory_omit_w = beta_x + beta_w * rho_xw / var_x

    summary = pd.DataFrame(
        [
            {"Estimator": "Truth", "Estimate": beta_x},
            {"Estimator": "Theory: OLS omit W", "Estimate": theory_omit_w},
            {"Estimator": "OLS omit W", "Estimate": _slope_with_intercept(x, y)},
            {"Estimator": "OLS control for W", "Estimate": _multiple_ols_coef(y, np.column_stack([x, w]), 0)},
            {"Estimator": "IV using Z", "Estimate": _iv_ratio(y, x, z)},
            {"Estimator": "corr(X, W)", "Estimate": np.corrcoef(x, w)[0, 1]},
        ]
    )
    return data, summary


def plot_ovb_sample(data: pd.DataFrame, beta_x: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = data["x"].to_numpy()
    w = data["w"].to_numpy()
    y = data["y"].to_numpy()

    axes[0].scatter(w, x, alpha=0.4, s=18, color="#55a868")
    axes[0].set_title("The omitted variable W is correlated with X")
    axes[0].set_xlabel("W (unobserved in the short regression)")
    axes[0].set_ylabel("X")
    axes[0].text(
        0.03,
        0.95,
        f"corr(X, W) = {np.corrcoef(x, w)[0, 1]:.3f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    short_fit = sm.OLS(y, sm.add_constant(x)).fit()
    long_fit = sm.OLS(y, sm.add_constant(np.column_stack([x, w]))).fit()
    x_grid = np.linspace(x.min(), x.max(), 200)

    axes[1].scatter(x, y, alpha=0.35, s=18, color="#4c72b0")
    axes[1].plot(x_grid, short_fit.params[0] + short_fit.params[1] * x_grid, color="#c44e52", linewidth=2, label="OLS omit W")
    axes[1].plot(x_grid, long_fit.params[0] + long_fit.params[1] * x_grid, color="#55a868", linewidth=2, label="OLS control W")
    axes[1].plot(x_grid, beta_x * x_grid, color="black", linestyle="--", linewidth=2, label="True slope")
    axes[1].set_title("Short regression versus the truth")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].legend(frameon=False)

    plt.tight_layout()
    plt.show()


def mc_ovb(
    n: int = 400,
    beta_x: float = 1.0,
    beta_w: float = 1.5,
    rho_xw: float = 0.7,
    instrument_strength: float = 0.9,
    x_noise_sd: float = 1.0,
    reps: int = 800,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    ols_short = np.empty(reps)
    ols_long = np.empty(reps)
    iv_est = np.empty(reps)

    for rep in range(reps):
        z = rng.normal(size=n)
        w = rng.normal(size=n)
        v = rng.normal(scale=x_noise_sd, size=n)
        eps = rng.normal(size=n)
        x = instrument_strength * z + rho_xw * w + v
        y = beta_x * x + beta_w * w + eps
        ols_short[rep] = _slope_with_intercept(x, y)
        ols_long[rep] = _multiple_ols_coef(y, np.column_stack([x, w]), 0)
        iv_est[rep] = _iv_ratio(y, x, z)

    return {
        "OLS omit W": ols_short,
        "OLS control W": ols_long,
        "IV using Z": iv_est,
    }


def sample_measurement_error(
    n: int = 400,
    beta_x: float = 1.0,
    instrument_strength: float = 0.9,
    latent_noise_sd: float = 1.0,
    measurement_noise_sd: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = _rng(seed)
    z = rng.normal(size=n)
    latent_noise = rng.normal(scale=latent_noise_sd, size=n)
    x_true = instrument_strength * z + latent_noise
    measurement_error = rng.normal(scale=measurement_noise_sd, size=n)
    eps = rng.normal(size=n)
    x_observed = x_true + measurement_error
    y = beta_x * x_true + eps
    return pd.DataFrame(
        {
            "z": z,
            "x_true": x_true,
            "x_observed": x_observed,
            "measurement_error": measurement_error,
            "y": y,
        }
    )


def measurement_error_one_run(
    n: int = 400,
    beta_x: float = 1.0,
    instrument_strength: float = 0.9,
    latent_noise_sd: float = 1.0,
    measurement_noise_sd: float = 1.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = sample_measurement_error(
        n=n,
        beta_x=beta_x,
        instrument_strength=instrument_strength,
        latent_noise_sd=latent_noise_sd,
        measurement_noise_sd=measurement_noise_sd,
        seed=seed,
    )

    x_true = data["x_true"].to_numpy()
    x_observed = data["x_observed"].to_numpy()
    y = data["y"].to_numpy()
    z = data["z"].to_numpy()
    theory = beta_x * np.var(x_true, ddof=1) / (np.var(x_true, ddof=1) + measurement_noise_sd**2)

    summary = pd.DataFrame(
        [
            {"Estimator": "Truth", "Estimate": beta_x},
            {"Estimator": "Theory: OLS with noisy X", "Estimate": theory},
            {"Estimator": "OLS using noisy X", "Estimate": _slope_with_intercept(x_observed, y)},
            {"Estimator": "OLS using true X*", "Estimate": _slope_with_intercept(x_true, y)},
            {"Estimator": "IV using Z", "Estimate": _iv_ratio(y, x_observed, z)},
            {"Estimator": "corr(X observed, measurement error)", "Estimate": np.corrcoef(x_observed, data["measurement_error"])[0, 1]},
        ]
    )
    return data, summary


def plot_measurement_error_sample(data: pd.DataFrame, beta_x: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x_true = data["x_true"].to_numpy()
    x_observed = data["x_observed"].to_numpy()
    y = data["y"].to_numpy()
    x_grid = np.linspace(min(x_true.min(), x_observed.min()), max(x_true.max(), x_observed.max()), 200)

    axes[0].scatter(x_true, x_observed, alpha=0.4, s=18, color="#4c72b0")
    axes[0].plot(x_grid, x_grid, color="black", linestyle="--", linewidth=2)
    axes[0].set_title("Observed X is a noisy measure of true X*")
    axes[0].set_xlabel("True X*")
    axes[0].set_ylabel("Observed X")

    noisy_fit = sm.OLS(y, sm.add_constant(x_observed)).fit()
    true_fit = sm.OLS(y, sm.add_constant(x_true)).fit()
    axes[1].scatter(x_observed, y, alpha=0.25, s=18, color="#c44e52", label="Observed X")
    axes[1].scatter(x_true, y, alpha=0.25, s=18, color="#55a868", label="True X*")
    axes[1].plot(x_grid, noisy_fit.params[0] + noisy_fit.params[1] * x_grid, color="#c44e52", linewidth=2, label="OLS using noisy X")
    axes[1].plot(x_grid, true_fit.params[0] + true_fit.params[1] * x_grid, color="#55a868", linewidth=2, label="OLS using true X*")
    axes[1].plot(x_grid, beta_x * x_grid, color="black", linestyle="--", linewidth=2, label="True slope")
    axes[1].set_title("Measurement error flattens the fitted line")
    axes[1].set_xlabel("Regressor value")
    axes[1].set_ylabel("Y")
    axes[1].legend(frameon=False)

    plt.tight_layout()
    plt.show()


def mc_measurement_error(
    n: int = 400,
    beta_x: float = 1.0,
    instrument_strength: float = 0.9,
    latent_noise_sd: float = 1.0,
    measurement_noise_sd: float = 1.0,
    reps: int = 800,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    ols_noisy = np.empty(reps)
    ols_true = np.empty(reps)
    iv_est = np.empty(reps)

    for rep in range(reps):
        z = rng.normal(size=n)
        latent_noise = rng.normal(scale=latent_noise_sd, size=n)
        x_true = instrument_strength * z + latent_noise
        measurement_error = rng.normal(scale=measurement_noise_sd, size=n)
        eps = rng.normal(size=n)
        x_observed = x_true + measurement_error
        y = beta_x * x_true + eps
        ols_noisy[rep] = _slope_with_intercept(x_observed, y)
        ols_true[rep] = _slope_with_intercept(x_true, y)
        iv_est[rep] = _iv_ratio(y, x_observed, z)

    return {
        "OLS using noisy X": ols_noisy,
        "OLS using true X*": ols_true,
        "IV using Z": iv_est,
    }


def sample_simultaneity(
    n: int = 400,
    demand_intercept: float = 10.0,
    demand_slope: float = -0.8,
    supply_intercept: float = 2.0,
    supply_slope: float = 1.2,
    instrument_strength: float = 1.5,
    demand_shock_sd: float = 1.0,
    supply_shock_sd: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = _rng(seed)
    z = rng.normal(size=n)
    u_d = rng.normal(scale=demand_shock_sd, size=n)
    u_s = rng.normal(scale=supply_shock_sd, size=n)

    price = (
        supply_intercept
        - demand_intercept
        + instrument_strength * z
        + u_s
        - u_d
    ) / (demand_slope - supply_slope)
    quantity = demand_intercept + demand_slope * price + u_d

    return pd.DataFrame({"z": z, "price": price, "quantity": quantity, "u_d": u_d, "u_s": u_s})


def simultaneity_one_run(
    n: int = 400,
    demand_intercept: float = 10.0,
    demand_slope: float = -0.8,
    supply_intercept: float = 2.0,
    supply_slope: float = 1.2,
    instrument_strength: float = 1.5,
    demand_shock_sd: float = 1.0,
    supply_shock_sd: float = 1.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = sample_simultaneity(
        n=n,
        demand_intercept=demand_intercept,
        demand_slope=demand_slope,
        supply_intercept=supply_intercept,
        supply_slope=supply_slope,
        instrument_strength=instrument_strength,
        demand_shock_sd=demand_shock_sd,
        supply_shock_sd=supply_shock_sd,
        seed=seed,
    )

    price = data["price"].to_numpy()
    quantity = data["quantity"].to_numpy()
    z = data["z"].to_numpy()
    u_d = data["u_d"].to_numpy()

    summary = pd.DataFrame(
        [
            {"Estimator": "Truth: demand slope", "Estimate": demand_slope},
            {"Estimator": "OLS of Q on P", "Estimate": _slope_with_intercept(price, quantity)},
            {"Estimator": "IV using supply shifter Z", "Estimate": _iv_ratio(quantity, price, z)},
            {"Estimator": "corr(P, demand shock)", "Estimate": np.corrcoef(price, u_d)[0, 1]},
            {"Estimator": "corr(Z, P)", "Estimate": np.corrcoef(z, price)[0, 1]},
        ]
    )
    return data, summary


def plot_supply_demand_worlds(
    demand_intercept: float = 10.0,
    demand_slope: float = -0.8,
    supply_intercept: float = 2.0,
    supply_slope: float = 1.2,
    n_points: int = 30,
    shock_sd: float = 2.0,
    seed: int | None = None,
) -> None:
    rng = _rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    q_grid = np.linspace(2, 14, 200)

    def price_from_demand(quantity: np.ndarray, demand_shift: float) -> np.ndarray:
        return (quantity - demand_intercept - demand_shift) / demand_slope

    def price_from_supply(quantity: np.ndarray, supply_shift: float) -> np.ndarray:
        return (quantity - supply_intercept - supply_shift) / supply_slope

    demand_shocks = rng.normal(scale=shock_sd, size=n_points)
    supply_shocks = rng.normal(scale=shock_sd, size=n_points)

    price_both = (supply_intercept - demand_intercept + supply_shocks - demand_shocks) / (demand_slope - supply_slope)
    quantity_both = demand_intercept + demand_slope * price_both + demand_shocks

    axes[0].scatter(quantity_both, price_both, color="#c44e52", alpha=0.7)
    axes[0].plot(q_grid, price_from_demand(q_grid, 0.0), color="#4c72b0", linewidth=2, label="Demand")
    axes[0].plot(q_grid, price_from_supply(q_grid, 0.0), color="#55a868", linewidth=2, label="Supply")
    axes[0].set_title("Both curves shift")
    axes[0].set_xlabel("Quantity")
    axes[0].set_ylabel("Price")
    axes[0].legend(frameon=False)

    supply_only = rng.normal(scale=shock_sd, size=n_points)
    price_supply = (supply_intercept - demand_intercept + supply_only) / (demand_slope - supply_slope)
    quantity_supply = demand_intercept + demand_slope * price_supply
    axes[1].scatter(quantity_supply, price_supply, color="#4c72b0", alpha=0.7)
    axes[1].plot(q_grid, price_from_demand(q_grid, 0.0), color="#4c72b0", linewidth=2, label="Demand")
    axes[1].plot(q_grid, price_from_supply(q_grid, 0.0), color="#55a868", linewidth=2, label="Supply")
    axes[1].set_title("Only supply shifts")
    axes[1].set_xlabel("Quantity")
    axes[1].set_ylabel("Price")
    axes[1].legend(frameon=False)

    demand_only = rng.normal(scale=shock_sd, size=n_points)
    price_demand = (supply_intercept - demand_intercept - demand_only) / (demand_slope - supply_slope)
    quantity_demand = supply_intercept + supply_slope * price_demand
    axes[2].scatter(quantity_demand, price_demand, color="#55a868", alpha=0.7)
    axes[2].plot(q_grid, price_from_demand(q_grid, 0.0), color="#4c72b0", linewidth=2, label="Demand")
    axes[2].plot(q_grid, price_from_supply(q_grid, 0.0), color="#55a868", linewidth=2, label="Supply")
    axes[2].set_title("Only demand shifts")
    axes[2].set_xlabel("Quantity")
    axes[2].set_ylabel("Price")
    axes[2].legend(frameon=False)

    fig.suptitle("Why simultaneity is different from omitted-variable bias", y=1.03)
    plt.tight_layout()
    plt.show()


def mc_simultaneity(
    n: int = 400,
    demand_intercept: float = 10.0,
    demand_slope: float = -0.8,
    supply_intercept: float = 2.0,
    supply_slope: float = 1.2,
    instrument_strength: float = 1.5,
    demand_shock_sd: float = 1.0,
    supply_shock_sd: float = 1.0,
    reps: int = 800,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    rng = _rng(seed)
    ols_est = np.empty(reps)
    iv_est = np.empty(reps)

    for rep in range(reps):
        z = rng.normal(size=n)
        u_d = rng.normal(scale=demand_shock_sd, size=n)
        u_s = rng.normal(scale=supply_shock_sd, size=n)
        price = (
            supply_intercept
            - demand_intercept
            + instrument_strength * z
            + u_s
            - u_d
        ) / (demand_slope - supply_slope)
        quantity = demand_intercept + demand_slope * price + u_d
        ols_est[rep] = _slope_with_intercept(price, quantity)
        iv_est[rep] = _iv_ratio(quantity, price, z)

    return {
        "OLS of Q on P": ols_est,
        "IV using supply shifter Z": iv_est,
    }
