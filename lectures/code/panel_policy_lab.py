from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from statsmodels.datasets import get_rdataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "legend.frameon": False,
        }
    )


def load_r_dataset(dataset: str, package: str) -> pd.DataFrame:
    try:
        data = get_rdataset(dataset, package).data.copy()
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the R dataset {package}::{dataset}. "
            "These notebooks use statsmodels.get_rdataset(), so the first run "
            "usually needs internet access unless the dataset is already cached."
        ) from exc
    return data


def _read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if len(df.columns) == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(path, sep=";")
        return df
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _rename_using_aliases(
    df: pd.DataFrame, aliases: dict[str, list[str]]
) -> pd.DataFrame:
    normalized = {_normalize_name(col): col for col in df.columns}
    rename_map: dict[str, str] = {}
    for canonical, options in aliases.items():
        if canonical in df.columns:
            continue
        for option in [canonical] + options:
            actual = normalized.get(_normalize_name(option))
            if actual is not None:
                rename_map[actual] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _load_local_dataset(candidates: Iterable[str]) -> pd.DataFrame:
    for filename in candidates:
        path = DATA_DIR / filename
        if path.exists():
            return _read_tabular_file(path)
    raise FileNotFoundError


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def prepare_fatalities_panel() -> pd.DataFrame:
    try:
        panel = _load_local_dataset(
            ["Fatalities.csv", "AER_Fatalities.csv", "fatalities.csv", "fatalities.dta"]
        )
    except FileNotFoundError:
        panel = load_r_dataset("Fatalities", "AER")

    panel = _rename_using_aliases(
        panel,
        {
            "state": ["State"],
            "year": ["Year"],
            "fatal1820": ["fatal_1820"],
            "pop1820": ["pop_1820"],
            "drinkage": ["mlda", "minimum_legal_drinking_age"],
        },
    )
    _require_columns(
        panel, ["state", "year", "fatal1820", "pop1820", "drinkage"], "Fatalities data"
    )
    panel["state"] = panel["state"].astype(str)
    panel["year"] = panel["year"].astype(int)
    panel["fatal_rate_1820"] = 100000.0 * panel["fatal1820"] / panel["pop1820"]
    panel["mlda21"] = (panel["drinkage"] == 21).astype(int)

    treated_year = (
        panel.loc[panel["mlda21"] == 1, ["state", "year"]]
        .groupby("state", as_index=False)["year"]
        .min()
        .rename(columns={"year": "adopt_year"})
    )
    panel = panel.merge(treated_year, on="state", how="left")
    panel["ever_treated"] = panel["adopt_year"].notna()
    panel["event_time"] = panel["year"] - panel["adopt_year"]
    return panel.sort_values(["state", "year"]).reset_index(drop=True)


def twfe_did(panel: pd.DataFrame, controls: Iterable[str] | None = None):
    controls = list(controls or [])
    rhs = ["mlda21"] + controls + ["C(state)", "C(year)"]
    formula = "fatal_rate_1820 ~ " + " + ".join(rhs)
    fit = smf.ols(formula, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["state"]}
    )
    return fit


def event_study_did(
    panel: pd.DataFrame,
    leads: int = 4,
    lags: int = 4,
    controls: Iterable[str] | None = None,
) -> pd.DataFrame:
    controls = list(controls or [])
    df = panel.copy()
    event_names = []
    event_map = []

    for k in range(-leads, lags + 1):
        if k == -1:
            continue
        if k < 0:
            name = f"lead_{abs(k)}"
        else:
            name = f"lag_{k}"
        df[name] = ((df["event_time"] == k) & df["ever_treated"]).astype(int)
        event_names.append(name)
        event_map.append((name, k))

    rhs = event_names + controls + ["C(state)", "C(year)"]
    formula = "fatal_rate_1820 ~ " + " + ".join(rhs)
    fit = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["state"]}
    )

    rows = []
    for name, k in event_map:
        estimate = fit.params.get(name, np.nan)
        se = fit.bse.get(name, np.nan)
        rows.append(
            {
                "term": name,
                "event_time": k,
                "estimate": estimate,
                "std_error": se,
                "ci_low": estimate - 1.96 * se,
                "ci_high": estimate + 1.96 * se,
            }
        )
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


def plot_did_group_means(panel: pd.DataFrame) -> None:
    grouped = (
        panel.groupby(["year", "mlda21"], as_index=False)["fatal_rate_1820"]
        .mean()
        .rename(columns={"mlda21": "treated_now"})
    )
    label_map = {0: "States with MLDA < 21", 1: "States with MLDA = 21"}
    grouped["group"] = grouped["treated_now"].map(label_map)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=grouped,
        x="year",
        y="fatal_rate_1820",
        hue="group",
        marker="o",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title("Youth traffic fatalities by current treatment status")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fatalities per 100,000 ages 18-20")
    ax.legend(title="")
    plt.tight_layout()


def plot_event_study(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(-0.5, color="gray", linestyle=":", linewidth=1.5)
    ax.errorbar(
        results["event_time"],
        results["estimate"],
        yerr=1.96 * results["std_error"],
        fmt="o-",
        color="#C44E52",
        ecolor="#4C72B0",
        capsize=4,
        linewidth=2,
    )
    ax.set_title("Event-study coefficients around MLDA 21 adoption")
    ax.set_xlabel("Event time")
    ax.set_ylabel("Coefficient relative to event time -1")
    plt.tight_layout()


def load_california_prop99() -> pd.DataFrame:
    try:
        df = _load_local_dataset(
            [
                "california_prop99.csv",
                "California_Prop99.csv",
                "prop99.csv",
                "smoking.csv",
                "california_prop99.dta",
                "prop99.dta",
            ]
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find a local California synthetic-control dataset. "
            "Place one of these files in lectures/code/data/: "
            "california_prop99.csv, prop99.csv, or smoking.csv."
        ) from exc

    df = _rename_using_aliases(
        df,
        {
            "State": ["state"],
            "Year": ["year"],
            "PacksPerCapita": [
                "packs_per_capita",
                "packspercapita",
                "cigsale",
                "cigsales",
            ],
        },
    )
    _require_columns(df, ["State", "Year", "PacksPerCapita"], "California data")
    df["State"] = df["State"].astype(str)
    df["Year"] = df["Year"].astype(int)
    return df


def load_germany_reunification() -> pd.DataFrame:
    try:
        df = _load_local_dataset(
            [
                "scpi_germany.csv",
                "germany_reunification.csv",
                "germany.csv",
                "scpi_germany.dta",
            ]
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find a local German reunification dataset. "
            "Place scpi_germany.csv or germany_reunification.csv in lectures/code/data/."
        ) from exc

    df = _rename_using_aliases(
        df,
        {
            "country": ["Country", "unit"],
            "year": ["Year", "time"],
            "gdp": ["GDP", "gdp_per_capita", "gdpcap"],
            "trade": ["Trade"],
            "schooling": ["Schooling", "education"],
            "industry": ["Industry", "industryshare"],
            "infrate": ["Inflation", "inflation", "inflate"],
        },
    )
    _require_columns(df, ["country", "year", "gdp"], "Germany data")
    df["country"] = df["country"].astype(str)
    df["year"] = df["year"].astype(int)
    return df


def _standardize_rows(matrix: np.ndarray) -> np.ndarray:
    out = matrix.astype(float).copy()
    for i in range(out.shape[0]):
        row = out[i]
        scale = row.std(ddof=0)
        if np.isclose(scale, 0.0):
            scale = 1.0
        out[i] = (row - row.mean()) / scale
    return out


@dataclass
class SyntheticControlResult:
    treated_unit: str
    treatment_start: int
    years: np.ndarray
    actual: np.ndarray
    synthetic: np.ndarray
    gap: np.ndarray
    weights: pd.DataFrame
    optimization_success: bool
    objective_value: float


def fit_synthetic_control(
    panel: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_unit: str,
    treatment_start: int,
    feature_cols: Iterable[str] | None = None,
) -> SyntheticControlResult:
    feature_cols = list(feature_cols or [])
    df = panel.copy()
    df = df.sort_values([unit_col, time_col]).reset_index(drop=True)

    wide_y = df.pivot(index=time_col, columns=unit_col, values=outcome_col).sort_index()
    if treated_unit not in wide_y.columns:
        raise ValueError(f"Treated unit '{treated_unit}' not found in panel.")

    donor_units = [col for col in wide_y.columns if col != treated_unit]
    pre_years = wide_y.index[wide_y.index < treatment_start]
    years = wide_y.index.to_numpy()

    y_treated = wide_y[treated_unit].to_numpy()
    feature_blocks = [wide_y.loc[pre_years, [treated_unit] + donor_units].to_numpy()]

    if feature_cols:
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        grouped = (
            df[df[time_col] < treatment_start]
            .groupby(unit_col)[feature_cols]
            .mean()
            .reindex([treated_unit] + donor_units)
        )
        feature_blocks.append(grouped.to_numpy().T)

    X = np.vstack(feature_blocks)
    X = _standardize_rows(X)
    x_treated = X[:, 0]
    X_donors = X[:, 1:]

    n_donors = len(donor_units)

    def objective(w: np.ndarray) -> float:
        resid = x_treated - X_donors @ w
        return float(resid @ resid)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n_donors)]
    w0 = np.full(n_donors, 1.0 / n_donors)

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x
    synthetic = wide_y[donor_units].to_numpy() @ weights
    gap = y_treated - synthetic

    weights_df = (
        pd.DataFrame({"unit": donor_units, "weight": weights})
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    return SyntheticControlResult(
        treated_unit=treated_unit,
        treatment_start=treatment_start,
        years=years,
        actual=y_treated,
        synthetic=synthetic,
        gap=gap,
        weights=weights_df,
        optimization_success=bool(result.success),
        objective_value=float(result.fun),
    )


def plot_donor_spaghetti(
    panel: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_unit: str,
    treatment_start: int,
) -> None:
    fig, ax = plt.subplots()
    for unit, unit_df in panel.groupby(unit_col):
        unit_df = unit_df.sort_values(time_col)
        if unit == treated_unit:
            ax.plot(
                unit_df[time_col],
                unit_df[outcome_col],
                color="#C44E52",
                linewidth=3,
                label=treated_unit,
                zorder=3,
            )
        else:
            ax.plot(
                unit_df[time_col],
                unit_df[outcome_col],
                color="lightgray",
                linewidth=1,
                alpha=0.8,
                zorder=1,
            )
    ax.axvline(treatment_start, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Treated unit against donor trajectories")
    ax.set_xlabel("Year")
    ax.set_ylabel(outcome_col)
    ax.legend()
    plt.tight_layout()


def plot_synth_paths(result: SyntheticControlResult, outcome_label: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(result.years, result.actual, color="#C44E52", linewidth=3, label=result.treated_unit)
    ax.plot(result.years, result.synthetic, color="#4C72B0", linewidth=3, linestyle="--", label="Synthetic control")
    ax.axvline(result.treatment_start, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Treated path versus synthetic path")
    ax.set_xlabel("Year")
    ax.set_ylabel(outcome_label)
    ax.legend()
    plt.tight_layout()


def plot_synth_gap(result: SyntheticControlResult, outcome_label: str) -> None:
    fig, ax = plt.subplots()
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(result.treatment_start, color="black", linestyle="--", linewidth=1.5)
    ax.plot(result.years, result.gap, color="#55A868", linewidth=3)
    ax.set_title("Treatment effect gap: actual minus synthetic")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Gap in {outcome_label}")
    plt.tight_layout()


def plot_top_weights(result: SyntheticControlResult, top_n: int = 10) -> None:
    top = result.weights.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(top))))
    ax.barh(top["unit"], top["weight"], color="#8172B3")
    ax.set_title("Largest donor weights")
    ax.set_xlabel("Weight")
    ax.set_ylabel("")
    plt.tight_layout()
