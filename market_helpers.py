"""
market_helpers.py

Helper functions for the Market Settlement Game (MSG) baseline valuation model.

Author: OpenAI
Date: 2026-04-19

Modeling notes:
- This is the "Option A" baseline model:
    * One unified market
    * All 7 portfolios included in the market stack
    * Units bid at marginal cost (plus an optional small adder)
    * Demand is stochastic with Normal forecast error (sigma = 6%)
    * Renewable availability is stochastic by technology and hour
- Startup and fixed daily O&M costs are NOT included in the bid price.
  They are accounted for later in profit calculations.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import math


# ------------------------------------------------------------
# Basic configuration pulled from the assignment
# ------------------------------------------------------------

WEST_LOAD_FORECAST = {1: 5500.0, 2: 3000.0, 3: 2700.0, 4: 8100.0}
EAST_LOAD_FORECAST = {1: 4500.0, 2: 2000.0, 3: 6500.0, 4: 4200.0}

UNIFIED_LOAD_FORECAST = {
    hour: WEST_LOAD_FORECAST[hour] + EAST_LOAD_FORECAST[hour]
    for hour in range(1, 5)
}

DEMAND_SIGMA_FRAC = 0.06

SOLAR_MEAN = {1: 0.45, 2: 0.95, 3: 0.75, 4: 0.25}
SOLAR_STD = {1: 0.06, 2: 0.06, 3: 0.06, 4: 0.06}

WAVE_MEAN = 0.90
WAVE_STD = 0.06

ONSHORE_WIND_WEIBULL_ALPHA = 1.8
ONSHORE_WIND_WEIBULL_BETA = 0.3

OFFSHORE_WIND_WEIBULL_ALPHA = 1.8
OFFSHORE_WIND_WEIBULL_BETA = 0.7

RENEWABLE_TYPES = {"Solar", "Wave", "Onshore Wind", "Offshore Wind"}
DISPATCHABLE_ALWAYS_AVAILABLE = {"Hydro", "Geothermal", "Natural Gas", "Nuclear", "Coal"}


# ------------------------------------------------------------
# Data loading and cleanup
# ------------------------------------------------------------

def clean_portfolio_dataframe(xlsx_path: str, sheet_name: str = "Portfolios") -> pd.DataFrame:
    """
    Read and clean the portfolio sheet from the uploaded Excel workbook.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name).copy()

    rename_map = {
        "Utility\nName": "utility_name",
        "Utility\nNumber": "utility_number",
        "Unit ID": "unit_id",
        "Location": "location",
        "Unit Type": "unit_type",
        "Max Capacity \n(MW)": "max_capacity_mw",
        "Avg. Capacity\nFactor": "avg_capacity_factor",
        "Fuel Cost\n($/MW-hr)": "fuel_cost",
        "Variable O&M Costs\n($/MW-hr)": "variable_om_cost",
        "Total Incremental Cost\n($/MW-hr)": "incremental_cost",
        "Fixed Daily O&M Costs\n($/day)": "fixed_daily_om_cost",
        "Start-up Costs\n($/start-up)": "startup_cost",
    }
    df = df.rename(columns=rename_map)

    # Forward fill utility names so every row has the portfolio label.
    df["utility_name"] = df["utility_name"].ffill()

    # Standard numeric cleanup.
    numeric_cols = [
        "utility_number",
        "unit_id",
        "max_capacity_mw",
        "avg_capacity_factor",
        "fuel_cost",
        "variable_om_cost",
        "incremental_cost",
        "fixed_daily_om_cost",
        "startup_cost",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop empty rows if any exist.
    df = df.dropna(subset=["utility_number", "unit_id", "unit_type", "max_capacity_mw"]).copy()

    # Make types tidy.
    df["utility_number"] = df["utility_number"].astype(int)
    df["unit_id"] = df["unit_id"].astype(int)
    df["utility_name"] = df["utility_name"].astype(str).str.strip()
    df["location"] = df["location"].astype(str).str.strip()
    df["unit_type"] = df["unit_type"].astype(str).str.strip()

    return df.reset_index(drop=True)


def portfolio_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize each portfolio's core characteristics.
    """
    summary = (
        df.groupby(["utility_number", "utility_name", "location"], as_index=False)
        .agg(
            total_nameplate_mw=("max_capacity_mw", "sum"),
            total_fixed_daily_om_cost=("fixed_daily_om_cost", "sum"),
            total_startups_if_all_units_start_once=("startup_cost", "sum"),
            average_incremental_cost=("incremental_cost", "mean"),
            num_units=("unit_id", "count"),
        )
        .sort_values("utility_number")
        .reset_index(drop=True)
    )
    return summary


# ------------------------------------------------------------
# Random sampling utilities
# ------------------------------------------------------------

def sample_demand_mw(hour: int, rng: np.random.Generator) -> float:
    """
    Sample unified demand for a given hour from a Normal distribution centered at forecast.
    Sigma = 6% of forecast per assignment.
    """
    mean = UNIFIED_LOAD_FORECAST[hour]
    sigma = DEMAND_SIGMA_FRAC * mean
    demand = rng.normal(loc=mean, scale=sigma)

    # Avoid pathological negative or near-zero demand values.
    return max(0.0, float(demand))


def _sample_weibull_scaled(alpha: float, beta_mean: float, rng: np.random.Generator) -> float:
    """
    Sample a Weibull random variable and scale it so that the expected value is beta_mean.

    numpy.weibull(alpha) gives a Weibull with scale = 1.
    Mean of Weibull(k=alpha, lambda=1) is Gamma(1 + 1/alpha).
    We choose lambda such that E[X] = beta_mean.
    """
    mean_unit_scale = float(math.gamma(1.0 + 1.0 / alpha))
    scale = beta_mean / mean_unit_scale
    value = scale * rng.weibull(alpha)
    return float(np.clip(value, 0.0, 1.0))


def sample_capacity_factor(unit_type: str, hour: int, rng: np.random.Generator) -> float:
    """
    Sample hourly capacity factor by technology.

    Important assignment simplification/interpretation:
    All resources of the same renewable technology have the same capacity factor
    within a given hour/day draw. This function samples one value per technology.
    """
    unit_type = unit_type.strip()

    if unit_type in DISPATCHABLE_ALWAYS_AVAILABLE:
        return 1.0

    if unit_type == "Solar":
        return float(np.clip(rng.normal(SOLAR_MEAN[hour], SOLAR_STD[hour]), 0.0, 1.0))

    if unit_type == "Wave":
        return float(np.clip(rng.normal(WAVE_MEAN, WAVE_STD), 0.0, 1.0))

    if unit_type == "Onshore Wind":
        return _sample_weibull_scaled(
            alpha=ONSHORE_WIND_WEIBULL_ALPHA,
            beta_mean=ONSHORE_WIND_WEIBULL_BETA,
            rng=rng,
        )

    if unit_type == "Offshore Wind":
        return _sample_weibull_scaled(
            alpha=OFFSHORE_WIND_WEIBULL_ALPHA,
            beta_mean=OFFSHORE_WIND_WEIBULL_BETA,
            rng=rng,
        )

    raise ValueError(f"Unrecognized unit type: {unit_type}")


def sample_hourly_technology_capacity_factors(hour: int, rng: np.random.Generator) -> Dict[str, float]:
    """
    Draw one capacity factor per technology for the hour.
    This matches the assignment statement that all resources of the same type
    share the same capacity factor in a given hour.
    """
    tech_cfs = {}
    for tech in sorted(RENEWABLE_TYPES):
        tech_cfs[tech] = sample_capacity_factor(tech, hour, rng)
    return tech_cfs


# ------------------------------------------------------------
# Market construction and clearing
# ------------------------------------------------------------

def build_hourly_supply_stack(
    df_units: pd.DataFrame,
    hour: int,
    tech_cfs: Dict[str, float],
    bid_adder: float = 0.0,
) -> pd.DataFrame:
    """
    Build the hourly supply stack for all units in the unified market.

    Returns a DataFrame with:
    - bid_price
    - available_capacity_mw
    - incremental_cost
    - utility info
    - unit info
    """
    stack = df_units.copy()

    def available_capacity(row: pd.Series) -> float:
        if row["unit_type"] in RENEWABLE_TYPES:
            cf = tech_cfs[row["unit_type"]]
            return float(row["max_capacity_mw"] * cf)
        return float(row["max_capacity_mw"])

    stack["available_capacity_mw"] = stack.apply(available_capacity, axis=1)
    stack["bid_price"] = stack["incremental_cost"] + float(bid_adder)

    # Sort by bid price, then by unit id for deterministic tie breaking.
    stack = stack.sort_values(["bid_price", "unit_id"]).reset_index(drop=True)
    return stack


def clear_uniform_price_market(stack: pd.DataFrame, demand_mw: float) -> Tuple[pd.DataFrame, float, bool]:
    """
    Clear a uniform-price market.

    Returns:
    - dispatch DataFrame with dispatched_mw per unit
    - market clearing price
    - blackout flag
    """
    dispatch = stack.copy()
    dispatch["dispatched_mw"] = 0.0

    total_available = float(dispatch["available_capacity_mw"].sum())
    if total_available + 1e-9 < demand_mw:
        # Blackout: by assignment, no one in that market gets revenue that hour.
        dispatch["dispatched_mw"] = 0.0
        return dispatch, np.nan, True

    remaining = float(demand_mw)
    clearing_price = 0.0

    for idx, row in dispatch.iterrows():
        avail = float(row["available_capacity_mw"])

        if remaining <= 1e-9:
            break

        if avail <= remaining + 1e-9:
            dispatch.at[idx, "dispatched_mw"] = avail
            remaining -= avail
            clearing_price = float(row["bid_price"])
        else:
            dispatch.at[idx, "dispatched_mw"] = remaining
            clearing_price = float(row["bid_price"])
            remaining = 0.0
            break

    return dispatch, clearing_price, False


# ------------------------------------------------------------
# Cost and revenue accounting
# ------------------------------------------------------------

def initialize_daily_portfolio_tracking(df_units: pd.DataFrame) -> pd.DataFrame:
    """
    Create one row per portfolio for day-level accounting.
    """
    portfolios = (
        df_units[["utility_number", "utility_name", "location"]]
        .drop_duplicates()
        .sort_values("utility_number")
        .reset_index(drop=True)
    )

    portfolios["revenue"] = 0.0
    portfolios["variable_cost"] = 0.0
    portfolios["startup_cost"] = 0.0
    portfolios["fixed_daily_om_cost"] = 0.0
    portfolios["blackout_penalty"] = 0.0
    portfolios["profit"] = 0.0
    return portfolios


def run_one_day_simulation(
    df_units: pd.DataFrame,
    rng: np.random.Generator,
    bid_adder: float = 0.0,
    blackout_penalty_per_portfolio: float = 10_000.0,
) -> Dict[str, pd.DataFrame]:
    """
    Run one 4-hour unified-market day.

    Returns a dictionary of detailed results:
    - hourly_market
    - hourly_unit_dispatch
    - daily_portfolio
    """
    units = df_units.copy()

    # Track whether each unit was producing in the previous hour.
    previous_on = {int(unit_id): False for unit_id in units["unit_id"].tolist()}

    hourly_market_rows: List[dict] = []
    hourly_dispatch_frames: List[pd.DataFrame] = []

    daily_portfolio = initialize_daily_portfolio_tracking(units)

    # Daily fixed O&M is incurred once per round regardless of dispatch.
    fixed_by_portfolio = (
    units.groupby(["utility_number", "utility_name"], as_index=False)[["fixed_daily_om_cost"]]
    .sum()
    .sort_values(by="utility_number")
    )
    daily_portfolio = daily_portfolio.merge(
        fixed_by_portfolio,
        on=["utility_number", "utility_name"],
        how="left",
        suffixes=("", "_x"),
    )
    daily_portfolio["fixed_daily_om_cost"] = daily_portfolio["fixed_daily_om_cost"].fillna(0.0)

    for hour in range(1, 5):
        demand_mw = sample_demand_mw(hour, rng)
        tech_cfs = sample_hourly_technology_capacity_factors(hour, rng)
        stack = build_hourly_supply_stack(units, hour, tech_cfs, bid_adder=bid_adder)

        dispatch, clearing_price, blackout = clear_uniform_price_market(stack, demand_mw)
        dispatch["hour"] = hour
        dispatch["market_clearing_price"] = clearing_price
        dispatch["demand_mw"] = demand_mw
        dispatch["blackout"] = blackout

        if blackout:
            # No revenue for anyone. Every portfolio receives the penalty.
            daily_portfolio["blackout_penalty"] += blackout_penalty_per_portfolio
        else:
            # Revenue = MCP * dispatched energy (1 hour interval => MWh = MW for the hour)
            dispatch["revenue"] = dispatch["dispatched_mw"] * clearing_price
            dispatch["variable_cost"] = dispatch["dispatched_mw"] * dispatch["incremental_cost"]

            # Startup cost occurs if the unit dispatches this hour after being off in previous hour.
            startup_costs = []
            current_on = {}
            for _, row in dispatch.iterrows():
                unit_id = int(row["unit_id"])
                on_now = float(row["dispatched_mw"]) > 1e-9
                current_on[unit_id] = on_now

                startup = float(row["startup_cost"]) if (on_now and not previous_on[unit_id]) else 0.0
                startup_costs.append(startup)

            dispatch["startup_cost_incurred"] = startup_costs

            # Update previous hour status.
            previous_on = current_on

            # Roll up hour-level costs/revenues to portfolio level.
            hour_portfolio = (
                dispatch.groupby(["utility_number", "utility_name"], as_index=False)
                .agg(
                    revenue=("revenue", "sum"),
                    variable_cost=("variable_cost", "sum"),
                    startup_cost=("startup_cost_incurred", "sum"),
                )
            )

            daily_portfolio = daily_portfolio.merge(
                hour_portfolio,
                on=["utility_number", "utility_name"],
                how="left",
                suffixes=("", "_hour"),
            )

            for col in ["revenue", "variable_cost", "startup_cost"]:
                hour_col = f"{col}_hour"
                if hour_col in daily_portfolio.columns:
                    daily_portfolio[col] = daily_portfolio[col] + daily_portfolio[hour_col].fillna(0.0)
                    daily_portfolio = daily_portfolio.drop(columns=[hour_col])

        hourly_market_rows.append(
            {
                "hour": hour,
                "forecast_demand_mw": UNIFIED_LOAD_FORECAST[hour],
                "sampled_demand_mw": demand_mw,
                "market_clearing_price": clearing_price,
                "blackout": blackout,
                "offshore_wind_cf": tech_cfs.get("Offshore Wind", np.nan),
                "onshore_wind_cf": tech_cfs.get("Onshore Wind", np.nan),
                "solar_cf": tech_cfs.get("Solar", np.nan),
                "wave_cf": tech_cfs.get("Wave", np.nan),
            }
        )

        hourly_dispatch_frames.append(dispatch)

    daily_portfolio["profit"] = (
        daily_portfolio["revenue"]
        - daily_portfolio["variable_cost"]
        - daily_portfolio["startup_cost"]
        - daily_portfolio["fixed_daily_om_cost"]
        - daily_portfolio["blackout_penalty"]
    )

    hourly_market = pd.DataFrame(hourly_market_rows)
    hourly_unit_dispatch = pd.concat(hourly_dispatch_frames, ignore_index=True)

    return {
        "hourly_market": hourly_market,
        "hourly_unit_dispatch": hourly_unit_dispatch,
        "daily_portfolio": daily_portfolio.sort_values("utility_number").reset_index(drop=True),
    }


def run_monte_carlo(
    df_units: pd.DataFrame,
    n_sims: int = 500,
    random_seed: int = 42,
    bid_adder: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple 4-hour day simulations.

    Returns summary and detailed simulation outputs.
    """
    rng = np.random.default_rng(random_seed)

    all_daily = []
    all_hourly_market = []
    all_dispatch = []

    for sim in range(1, n_sims + 1):
        sim_results = run_one_day_simulation(df_units=df_units, rng=rng, bid_adder=bid_adder)
        
        dispatch = collect_dispatch_results(sim_results, sim)
        all_dispatch.append(dispatch)
        
        daily = sim_results["daily_portfolio"].copy()
        daily["simulation"] = sim

        hourly_market = sim_results["hourly_market"].copy()
        hourly_market["simulation"] = sim

        all_daily.append(daily)
        all_hourly_market.append(hourly_market)

    all_daily_df = pd.concat(all_daily, ignore_index=True)
    all_hourly_market_df = pd.concat(all_hourly_market, ignore_index=True)
    all_dispatch_df = pd.concat(all_dispatch, ignore_index=True)

    portfolio_summary = (
        all_daily_df.groupby(["utility_number", "utility_name", "location"], as_index=False)
        .agg(
            expected_daily_revenue=("revenue", "mean"),
            revenue_std=("revenue", "std"),
            expected_daily_profit=("profit", "mean"),
            profit_std=("profit", "std"),
            expected_variable_cost=("variable_cost", "mean"),
            expected_startup_cost=("startup_cost", "mean"),
            fixed_daily_om_cost=("fixed_daily_om_cost", "mean"),
            expected_blackout_penalty=("blackout_penalty", "mean"),
        )
        .sort_values("utility_number")
        .reset_index(drop=True)
    )

    # Option A baseline: represent the 8-day game by 8 identical day draws.
    portfolio_summary["expected_8day_revenue"] = 8.0 * portfolio_summary["expected_daily_revenue"]
    portfolio_summary["expected_8day_profit"] = 8.0 * portfolio_summary["expected_daily_profit"]

    hourly_price_summary = (
        all_hourly_market_df.groupby("hour", as_index=False)
        .agg(
            expected_price=("market_clearing_price", "mean"),
            price_std=("market_clearing_price", "std"),
            expected_demand=("sampled_demand_mw", "mean"),
            blackout_rate=("blackout", "mean"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )

    return {
        "all_daily_results": all_daily_df,
        "all_hourly_market_results": all_hourly_market_df,
        "portfolio_summary": portfolio_summary,
        "hourly_price_summary": hourly_price_summary,
        "all_dispatch_results": all_dispatch_df,
    }


# ------------------------------------------------------------
# Deterministic reference scenarios for write-up support
# ------------------------------------------------------------

def build_average_availability_stack(df_units: pd.DataFrame, hour: int, bid_adder: float = 0.0) -> pd.DataFrame:
    """
    Build a deterministic stack using average renewable capacity factors.
    Useful for quick one-hour scenario analysis in the report.
    """
    avg_tech_cfs = {
        "Offshore Wind": OFFSHORE_WIND_WEIBULL_BETA,
        "Onshore Wind": ONSHORE_WIND_WEIBULL_BETA,
        "Solar": SOLAR_MEAN[hour],
        "Wave": WAVE_MEAN,
    }
    return build_hourly_supply_stack(df_units, hour, avg_tech_cfs, bid_adder=bid_adder)


def run_deterministic_hour_case(
    df_units: pd.DataFrame,
    demand_mw: float,
    hour: int = 1,
    bid_adder: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """
    Run one deterministic hour using average renewable output.

    This helps answer questions like:
    - What would each portfolio expect to make in a one-hour market if total load = 9000 MW?
    """
    stack = build_average_availability_stack(df_units, hour=hour, bid_adder=bid_adder)
    dispatch, clearing_price, blackout = clear_uniform_price_market(stack, demand_mw=demand_mw)

    if blackout:
        portfolio_hour = (
            dispatch.groupby(["utility_number", "utility_name", "location"], as_index=False)
            .agg(dispatched_mw=("dispatched_mw", "sum"))
            .sort_values("utility_number")
        )
        portfolio_hour["revenue"] = 0.0
    else:
        dispatch["revenue"] = dispatch["dispatched_mw"] * clearing_price
        portfolio_hour = (
            dispatch.groupby(["utility_number", "utility_name", "location"], as_index=False)
            .agg(
                dispatched_mw=("dispatched_mw", "sum"),
                revenue=("revenue", "sum"),
            )
            .sort_values("utility_number")
            .reset_index(drop=True)
        )

    return {
        "stack": stack,
        "dispatch": dispatch,
        "portfolio_hour": portfolio_hour,
        "market_info": pd.DataFrame(
            [{
                "hour": hour,
                "demand_mw": demand_mw,
                "market_clearing_price": clearing_price,
                "blackout": blackout,
            }]
        ),
    }

def collect_dispatch_results(sim_results, sim_number):
    dispatch = sim_results["hourly_unit_dispatch"].copy()
    dispatch["simulation"] = sim_number
    return dispatch