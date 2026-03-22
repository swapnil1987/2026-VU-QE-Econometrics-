from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent


def _cell_metadata(slide_type: str | None = None) -> dict:
    metadata = {}
    if slide_type is not None:
        metadata["slideshow"] = {"slide_type": slide_type}
    return metadata


def md(text: str, slide_type: str | None = None):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip(), metadata=_cell_metadata(slide_type))


SETUP_GUARD = """
if "lab" not in globals():
    raise RuntimeError("Run the setup cell on the previous slide first.")
"""


def code(text: str, slide_type: str | None = None, needs_setup: bool = False):
    source = textwrap.dedent(text).strip()
    if needs_setup:
        source = SETUP_GUARD.strip() + "\n\n" + source
    return nbf.v4.new_code_cell(source, metadata=_cell_metadata(slide_type))


IMPORTS = """
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

candidate_dirs = [Path.cwd(), Path.cwd() / "lectures" / "code"]
for candidate in candidate_dirs:
    if (candidate / "endogeneity_lab.py").exists():
        sys.path.insert(0, str(candidate))
        break

import endogeneity_lab as lab

lab.set_plot_style()
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
"""


def build_notebook_01() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # Endogeneity vs Exogeneity: Why OLS Sometimes Fails

            **Course**: Quantitative Econometrics  
            **Audience**: Bachelor students  
            **Instructor**: Swapnil Singh

            This notebook comes **before** the IV lecture. The goal is not to master IV yet. The goal is to see, in code, what the core problem is.

            ## Learning goals

            1. Understand the regression model $Y_i = \\beta_0 + \\beta_1 X_i + u_i$
            2. See why OLS needs $\\operatorname{cov}(X_i, u_i) = 0$
            3. See what changes when $X$ becomes correlated with the hidden error term $u$
            4. Build intuition for the three standard sources of endogeneity that we will study next
            """,
            slide_type="slide",
        ),
        md(
            r"""
            ## The central idea

            In a simple regression, $u_i$ collects **everything that affects $Y_i$ but is not written explicitly in the model**.

            If $X_i$ is unrelated to that hidden part, OLS has a chance:
            $$
            \operatorname{cov}(X_i, u_i) = 0.
            $$

            If $X_i$ is related to the hidden part, OLS mixes up two forces:

            - the genuine causal effect of $X$ on $Y$
            - the hidden information inside $u$

            That is **endogeneity**:
            $$
            \operatorname{cov}(X_i, u_i) \neq 0.
            $$
            """,
            slide_type="subslide",
        ),
        md(
            """
            ## Setup

            Run the next cell once at the start of the presentation. If the kernel restarts during RISE, come back and run it again.
            """,
            slide_type="slide",
        ),
        code(IMPORTS, slide_type="fragment"),
        md(
            """
            ## Step 1. A clean exogenous world

            In the next cell, `gamma = 0.0`. That means the regressor `X` is not built from the hidden error `u`.
            """,
            slide_type="slide",
        ),
        code(
            """
            beta_true = 1.0

            exogenous = lab.sample_basic_endogeneity(
                n=300,
                beta=beta_true,
                gamma=0.0,
                seed=2026,
            )

            display(lab.basic_case_summary(exogenous, beta_true))
            lab.plot_basic_case(exogenous, beta_true, "Exogenous case")
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            In the summary above, focus on two numbers:

            - `corr(X, u)`: it should be close to zero
            - `OLS slope`: it should be close to the true slope `beta = 1`

            In the right-hand figure we can literally see the hidden error `u` because this is a simulation. In real data we do **not** observe `u`, which is exactly why endogeneity is hard.
            """,
            slide_type="fragment",
        ),
        md(
            """
            ## Step 2. Now make the regressor endogenous

            In the next cell, `gamma = 0.9`. That means part of `X` is mechanically built from `u`.
            """,
            slide_type="slide",
        ),
        code(
            """
            endogenous = lab.sample_basic_endogeneity(
                n=300,
                beta=beta_true,
                gamma=0.9,
                seed=2026,
            )

            display(lab.basic_case_summary(endogenous, beta_true))
            lab.plot_basic_case(endogenous, beta_true, "Endogenous case")
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            The pictures should now look different:

            - `corr(X, u)` is no longer near zero
            - the OLS line moves away from the true line

            This is the whole problem in one sentence: **OLS is biased because the regressor carries hidden information from the error term.**
            """,
            slide_type="fragment",
        ),
        md(
            """
            ## Step 3. Monte Carlo: bias grows as endogeneity gets stronger

            One sample can be unlucky. So we repeat the same experiment many times and average the results.
            """,
            slide_type="slide",
        ),
        code(
            """
            gamma_grid = (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)

            mc_results = lab.mc_basic_endogeneity(
                n=300,
                beta=beta_true,
                gamma_values=gamma_grid,
                reps=600,
                seed=2026,
            )

            display(mc_results)
            lab.plot_basic_monte_carlo(mc_results, beta_true)
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            Read the table and figures as follows:

            - when `gamma = 0`, the average OLS estimate is close to the truth
            - when `gamma > 0`, OLS is biased upward
            - when `gamma < 0`, OLS is biased downward

            The sign of the bias depends on the sign of the correlation between `X` and `u`.
            """,
            slide_type="fragment",
        ),
        md(
            """
            ## What creates endogeneity in practice?

            The algebra is always the same: `cov(X, u) != 0`.

            But the **mechanism** can differ. In introductory econometrics, the three standard mechanisms are:

            1. **Omitted variable bias**: something important is left out of the regression, ends up inside `u`, and is correlated with `X`
            2. **Measurement error**: the regressor is measured with noise, so the noise leaks into both `X` and `u`
            3. **Simultaneity**: `X` and `Y` are jointly determined, so `X` already reflects shocks hitting `Y`

            Notebook 2 studies those three cases one by one.
            """,
            slide_type="slide",
        ),
        md(
            """
            ## Try it yourself

            The next cell is meant for students to edit. Change `trial_gamma`, rerun, and watch how the summary table and the figures move.
            """,
            slide_type="subslide",
        ),
        code(
            """
            trial_gamma = 0.4

            trial = lab.sample_basic_endogeneity(
                n=300,
                beta=beta_true,
                gamma=trial_gamma,
                seed=99,
            )

            display(lab.basic_case_summary(trial, beta_true))
            lab.plot_basic_case(trial, beta_true, f"Trial case with gamma = {trial_gamma}")
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            ## Takeaway

            If `X` is clean, OLS can recover the true slope. If `X` is contaminated by the hidden part of the outcome equation, OLS loses its causal interpretation.

            The next notebook answers the natural follow-up question: **what are the concrete economic stories that make `cov(X, u)` nonzero?**
            """,
            slide_type="slide",
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python"}
    notebook["metadata"]["livereveal"] = {
        "autolaunch": False,
        "controls": True,
        "progress": True,
        "scroll": True,
        "slideNumber": True,
        "transition": "none",
    }
    return notebook


def build_notebook_02() -> nbf.NotebookNode:
    cells = [
        md(
            """
            # Three Sources of Endogeneity

            **Course**: Quantitative Econometrics  
            **Audience**: Bachelor students  
            **Instructor**: Swapnil Singh

            This notebook is still **pre-IV**. We are learning the problem first. At the end of each section, we give a short preview of how an external source of variation could fix the problem, but the formal IV lecture comes later.

            ## Learning goals

            1. See three different ways to create the same problem: $\\operatorname{cov}(X_i, u_i) \\neq 0$
            2. Connect intuitive stories, algebra, and simulation output
            3. Learn which settings can be fixed with controls and which ones require something stronger
            """,
            slide_type="slide",
        ),
        md(
            r"""
            ## One framework, three mechanisms

            We keep returning to the same regression:
            $$
            Y_i = \beta_0 + \beta_1 X_i + u_i.
            $$

            OLS needs $X_i$ to be unrelated to the hidden term $u_i$. The three mechanisms below break that requirement in different ways:

            | Source | What goes wrong? | Why does $X$ become correlated with $u$? |
            |---|---|---|
            | Omitted variable | Something relevant is left out | The omitted factor affects both $X$ and $Y$ |
            | Measurement error | We observe a noisy version of $X$ | The noise contaminates both the regressor and the error |
            | Simultaneity | $X$ and $Y$ are jointly chosen | The regressor already responds to shocks in the outcome equation |
            """,
            slide_type="subslide",
        ),
        md(
            """
            ## Setup

            Run the next cell once at the start of the presentation. If the kernel restarts during RISE, come back and run it again.
            """,
            slide_type="slide",
        ),
        code(IMPORTS, slide_type="fragment"),
        code(
            """
            sample_size = 400
            mc_reps = 700
            """,
            slide_type="fragment",
        ),
        md(
            r"""
            ## 1. Omitted Variable Bias

            ### Intuition

            Suppose we want to estimate the effect of schooling on earnings, but we omit ability.

            - ability affects earnings directly
            - ability also affects schooling

            Then ability is hidden inside $u$, and the regressor `X` inherits part of it.

            ### Algebra

            True model:
            $$
            Y_i = \beta_0 + \beta_1 X_i + \beta_2 W_i + \varepsilon_i
            $$

            Estimated short regression:
            $$
            Y_i = \beta_0 + \beta_1 X_i + u_i, \quad u_i = \beta_2 W_i + \varepsilon_i
            $$

            Therefore,
            $$
            \operatorname{cov}(X_i, u_i) = \beta_2 \operatorname{cov}(X_i, W_i).
            $$

            Omitted variable bias disappears if either:

            - the omitted factor does not matter for $Y$ (`beta_w = 0`)
            - the omitted factor is unrelated to $X$ (`rho_xw = 0`)
            """,
            slide_type="slide",
        ),
        code(
            """
            ovb_params = {
                "n": sample_size,
                "beta_x": 1.0,
                "beta_w": 1.5,
                "rho_xw": 0.8,
                "instrument_strength": 0.9,
                "x_noise_sd": 1.0,
            }
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        code(
            """
            ovb_data, ovb_snapshot = lab.ovb_one_run(**ovb_params, seed=2026)

            display(ovb_snapshot)
            lab.plot_ovb_sample(ovb_data, beta_x=ovb_params["beta_x"])
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            Read the snapshot like this:

            - `OLS omit W` is the biased short regression
            - `OLS control for W` is the clean benchmark when the omitted variable becomes observable
            - `IV using Z` is only a preview for later: if we had a clean shifter `Z`, it could also isolate variation in `X` that is unrelated to `W`

            If you want to turn off omitted-variable bias and verify the theory, set `rho_xw = 0` or `beta_w = 0` and rerun the two cells above.
            """,
            slide_type="subslide",
        ),
        code(
            """
            ovb_mc = lab.mc_ovb(**ovb_params, reps=mc_reps, seed=1)
            ovb_summary = lab.summarize_estimates(ovb_mc, truth=ovb_params["beta_x"])

            display(ovb_summary)
            lab.plot_estimate_distributions(
                ovb_mc,
                truth=ovb_params["beta_x"],
                title="Omitted variable bias: the short regression drifts away from the truth",
                x_label="Estimated slope on X",
            )
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            r"""
            ## 2. Measurement Error

            ### Intuition

            Sometimes the regressor is conceptually right, but measured badly. Think of self-reported income, hours worked, or wealth.

            Let the true regressor be $X_i^\*$, but suppose we observe:
            $$
            X_i = X_i^\* + w_i.
            $$

            ### Algebra

            If the true model is
            $$
            Y_i = \beta_0 + \beta_1 X_i^\* + \varepsilon_i,
            $$
            then after substituting $X_i^\* = X_i - w_i$ we get
            $$
            Y_i = \beta_0 + \beta_1 X_i + (\varepsilon_i - \beta_1 w_i).
            $$

            The new error term contains the measurement noise. So the observed regressor and the error term now overlap mechanically.

            Under classical measurement error, OLS is biased toward zero:
            $$
            \hat{\beta}_1^{OLS} \to \beta_1 \frac{\operatorname{var}(X^\*)}{\operatorname{var}(X^\*) + \operatorname{var}(w)}.
            $$
            """,
            slide_type="slide",
        ),
        code(
            """
            measurement_params = {
                "n": sample_size,
                "beta_x": 1.0,
                "instrument_strength": 0.9,
                "latent_noise_sd": 1.0,
                "measurement_noise_sd": 1.1,
            }
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        code(
            """
            me_data, me_snapshot = lab.measurement_error_one_run(**measurement_params, seed=2026)

            display(me_snapshot)
            lab.plot_measurement_error_sample(me_data, beta_x=measurement_params["beta_x"])
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            Focus on the slope estimates:

            - `OLS using true X*` is the benchmark we would like to have
            - `OLS using noisy X` is flatter because measurement noise pushes the coefficient toward zero
            - `IV using Z` is a preview of how an external source of clean variation could recover the true signal

            To see attenuation bias disappear, set `measurement_noise_sd = 0` and rerun the last two cells.
            """,
            slide_type="subslide",
        ),
        code(
            """
            me_mc = lab.mc_measurement_error(**measurement_params, reps=mc_reps, seed=2)
            me_summary = lab.summarize_estimates(me_mc, truth=measurement_params["beta_x"])

            display(me_summary)
            lab.plot_estimate_distributions(
                me_mc,
                truth=measurement_params["beta_x"],
                title="Measurement error: OLS is pulled toward zero",
                x_label="Estimated slope on X",
            )
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            r"""
            ## 3. Simultaneity

            ### Intuition

            In supply and demand, price and quantity are determined together. If we run a regression of quantity on price, price is not an outside cause that arrives first. It is an equilibrium outcome.

            Demand:
            $$
            Q_i = \alpha_0 + \alpha_1 P_i + u_i^d
            $$

            Supply:
            $$
            Q_i = \gamma_0 + \gamma_1 P_i + \delta Z_i + u_i^s
            $$

            Here `Z` is a supply shifter. We use it only as a preview device for later.
            """,
            slide_type="slide",
        ),
        code(
            """
            lab.plot_supply_demand_worlds(seed=2026)
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            The figure above is the key intuition:

            - if both supply and demand move, the cloud of equilibrium points does not trace a single structural curve
            - if only supply shifts, the equilibrium points trace out demand
            - if only demand shifts, the equilibrium points trace out supply

            This is why simultaneity is not just another omitted-variable story. The regressor itself is jointly determined inside the system.
            """,
            slide_type="subslide",
        ),
        code(
            """
            simultaneity_params = {
                "n": sample_size,
                "demand_intercept": 10.0,
                "demand_slope": -0.8,
                "supply_intercept": 2.0,
                "supply_slope": 1.2,
                "instrument_strength": 1.5,
                "demand_shock_sd": 1.0,
                "supply_shock_sd": 1.0,
            }
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        code(
            """
            sim_data, sim_snapshot = lab.simultaneity_one_run(**simultaneity_params, seed=2026)

            display(sim_snapshot)
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            The important numbers are:

            - `corr(P, demand shock)`: this shows price is endogenous in the demand equation
            - `OLS of Q on P`: this does not recover the true demand slope
            - `IV using supply shifter Z`: preview of the idea that a variable shifting supply but not demand can identify the demand curve
            """,
            slide_type="subslide",
        ),
        code(
            """
            sim_mc = lab.mc_simultaneity(**simultaneity_params, reps=mc_reps, seed=3)
            sim_summary = lab.summarize_estimates(sim_mc, truth=simultaneity_params["demand_slope"])

            display(sim_summary)
            lab.plot_estimate_distributions(
                sim_mc,
                truth=simultaneity_params["demand_slope"],
                title="Simultaneity: OLS mixes supply and demand",
                x_label="Estimated demand slope",
                xlim=(-1.6, 1.2),
            )
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            ## Side-by-side comparison

            The three cases look different economically, but from the point of view of regression they all create the same symptom: `X` carries hidden information from the error term.
            """,
            slide_type="slide",
        ),
        code(
            """
            if "mc_reps" not in globals():
                mc_reps = 700

            if "ovb_params" not in globals():
                ovb_params = {
                    "n": 400,
                    "beta_x": 1.0,
                    "beta_w": 1.5,
                    "rho_xw": 0.8,
                    "instrument_strength": 0.9,
                    "x_noise_sd": 1.0,
                }
            if "ovb_summary" not in globals():
                ovb_mc = lab.mc_ovb(**ovb_params, reps=mc_reps, seed=1)
                ovb_summary = lab.summarize_estimates(ovb_mc, truth=ovb_params["beta_x"])

            if "measurement_params" not in globals():
                measurement_params = {
                    "n": 400,
                    "beta_x": 1.0,
                    "instrument_strength": 0.9,
                    "latent_noise_sd": 1.0,
                    "measurement_noise_sd": 1.1,
                }
            if "me_summary" not in globals():
                me_mc = lab.mc_measurement_error(**measurement_params, reps=mc_reps, seed=2)
                me_summary = lab.summarize_estimates(me_mc, truth=measurement_params["beta_x"])

            if "simultaneity_params" not in globals():
                simultaneity_params = {
                    "n": 400,
                    "demand_intercept": 10.0,
                    "demand_slope": -0.8,
                    "supply_intercept": 2.0,
                    "supply_slope": 1.2,
                    "instrument_strength": 1.5,
                    "demand_shock_sd": 1.0,
                    "supply_shock_sd": 1.0,
                }
            if "sim_summary" not in globals():
                sim_mc = lab.mc_simultaneity(**simultaneity_params, reps=mc_reps, seed=3)
                sim_summary = lab.summarize_estimates(sim_mc, truth=simultaneity_params["demand_slope"])

            comparison = pd.DataFrame(
                [
                    {
                        "Source": "Omitted variable bias",
                        "Biased OLS mean": ovb_summary.loc[ovb_summary["Estimator"] == "OLS omit W", "Mean"].iloc[0],
                        "Cleaner estimator mean": ovb_summary.loc[ovb_summary["Estimator"] == "IV using Z", "Mean"].iloc[0],
                        "Truth": ovb_params["beta_x"],
                    },
                    {
                        "Source": "Measurement error",
                        "Biased OLS mean": me_summary.loc[me_summary["Estimator"] == "OLS using noisy X", "Mean"].iloc[0],
                        "Cleaner estimator mean": me_summary.loc[me_summary["Estimator"] == "IV using Z", "Mean"].iloc[0],
                        "Truth": measurement_params["beta_x"],
                    },
                    {
                        "Source": "Simultaneity",
                        "Biased OLS mean": sim_summary.loc[sim_summary["Estimator"] == "OLS of Q on P", "Mean"].iloc[0],
                        "Cleaner estimator mean": sim_summary.loc[sim_summary["Estimator"] == "IV using supply shifter Z", "Mean"].iloc[0],
                        "Truth": simultaneity_params["demand_slope"],
                    },
                ]
            )

            display(comparison)
            """,
            slide_type="fragment",
            needs_setup=True,
        ),
        md(
            """
            ## Final takeaways

            1. Different economic stories can produce the same econometric symptom: `cov(X, u) != 0`
            2. Omitted variables can sometimes be fixed by adding the missing control, if it is observed
            3. Measurement error and simultaneity usually need a different source of variation
            4. That different source of variation is exactly what the IV lecture is about

            If you want a clean classroom exercise, rerun each section with the bias switched off:

            - OVB: set `rho_xw = 0`
            - Measurement error: set `measurement_noise_sd = 0`
            - Simultaneity: set `demand_shock_sd = 0` and inspect how the equilibrium cloud changes
            """,
            slide_type="slide",
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python"}
    notebook["metadata"]["livereveal"] = {
        "autolaunch": False,
        "controls": True,
        "progress": True,
        "scroll": True,
        "slideNumber": True,
        "transition": "none",
    }
    return notebook


def main() -> None:
    notebooks = {
        NOTEBOOK_DIR / "01_exogeneity_deep_dive.ipynb": build_notebook_01(),
        NOTEBOOK_DIR / "02_sources_of_endogeneity.ipynb": build_notebook_02(),
    }

    for path, notebook in notebooks.items():
        nbf.write(notebook, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
