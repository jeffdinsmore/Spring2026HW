"""
market_settlement_main.py

Main driver script for the Market Settlement Game (MSG) baseline valuation model.

Author: OpenAI
Date: 2026-04-19
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from market_helpers import (
    UNIFIED_LOAD_FORECAST,
    clean_portfolio_dataframe,
    portfolio_summary_table,
    run_deterministic_hour_case,
    run_monte_carlo,
)


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

# Update this path if your Excel file is stored somewhere else.
#BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = "Portfolios-3.xlsx"

# Number of Monte Carlo simulation days to run.
N_SIMS = 500

# Set to a small positive number later if you want to model
# "bid slightly above marginal cost" behavior.
BID_ADDER = 0.0

# Output folder for CSV summaries and plots.
OUTPUT_DIR = Path("msg_outputs")


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------

def main() -> None:
    # --------------------------------------------------------
    # Step 1: Create output folder.
    # --------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Step 2: Load and summarize generator data.
    # --------------------------------------------------------
    df_units = clean_portfolio_dataframe(EXCEL_FILE)
    unit_summary = portfolio_summary_table(df_units)

    print("\n" + "=" * 72)
    print("MSG BASELINE PORTFOLIO DATA SUMMARY")
    print("=" * 72)
    print(unit_summary.to_string(index=False))

    unit_summary.to_csv(OUTPUT_DIR / "portfolio_input_summary.csv", index=False)

    # --------------------------------------------------------
    # Step 3: Run Monte Carlo baseline for one representative day.
    # --------------------------------------------------------
    results = run_monte_carlo(
        df_units=df_units,
        n_sims=N_SIMS,
        random_seed=42,
        bid_adder=BID_ADDER,
    )

    portfolio_summary = results["portfolio_summary"].copy()
    hourly_price_summary = results["hourly_price_summary"].copy()
    all_daily = results["all_daily_results"].copy()

    # Save CSV outputs.
    portfolio_summary.to_csv(OUTPUT_DIR / "portfolio_summary_baseline.csv", index=False)
    hourly_price_summary.to_csv(OUTPUT_DIR / "hourly_price_summary_baseline.csv", index=False)
    all_daily.to_csv(OUTPUT_DIR / "all_daily_results_baseline.csv", index=False)

    print("\n" + "=" * 72)
    print("MONTE CARLO BASELINE RESULTS")
    print("=" * 72)
    print(portfolio_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("EXPECTED HOURLY PRICE SUMMARY")
    print("=" * 72)
    print(hourly_price_summary.to_string(index=False))

    # --------------------------------------------------------
    # Step 4: Deterministic reference case for the write-up.
    # --------------------------------------------------------
    # The assignment asks what each portfolio might make in one hour
    # if total system load was 9000 MW. We use average renewable output
    # for this deterministic benchmark case.
    det_9000 = run_deterministic_hour_case(df_units=df_units, demand_mw=9000.0, hour=1, bid_adder=BID_ADDER)

    det_9000["portfolio_hour"].to_csv(OUTPUT_DIR / "deterministic_9000mw_hour_case.csv", index=False)
    det_9000["market_info"].to_csv(OUTPUT_DIR / "deterministic_9000mw_market_info.csv", index=False)

    print("\n" + "=" * 72)
    print("DETERMINISTIC ONE-HOUR CASE: TOTAL LOAD = 9000 MW")
    print("=" * 72)
    print(det_9000["market_info"].to_string(index=False))
    print(det_9000["portfolio_hour"].to_string(index=False))

    # --------------------------------------------------------
    # Step 5: Make plots.
    # --------------------------------------------------------

    # Plot 1: Expected clearing price by hour.
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        hourly_price_summary["hour"],
        hourly_price_summary["expected_price"],
        yerr=hourly_price_summary["price_std"],
        marker="o",
        linestyle="-",
        capsize=4,
    )
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("Hour")
    plt.ylabel("Market Clearing Price ($/MWh)")
    plt.title("Expected Market Clearing Price by Hour")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expected_market_clearing_price_by_hour.png", dpi=200)
    plt.close()

    # Plot 2: Expected daily revenue by portfolio.
    plt.figure(figsize=(10, 6))
    plt.bar(portfolio_summary["utility_name"], portfolio_summary["expected_daily_revenue"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Expected Daily Revenue ($)")
    plt.title("Expected Daily Revenue by Portfolio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expected_daily_revenue_by_portfolio.png", dpi=200)
    plt.close()

    # Plot 3: Expected 8-day profit by portfolio.
    plt.figure(figsize=(10, 6))
    plt.bar(portfolio_summary["utility_name"], portfolio_summary["expected_8day_profit"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Expected 8-Day Profit ($)")
    plt.title("Expected 8-Day Profit by Portfolio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "expected_8day_profit_by_portfolio.png", dpi=200)
    plt.close()

    # Plot 4: Revenue variability by portfolio.
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        portfolio_summary["utility_name"],
        portfolio_summary["expected_daily_revenue"],
        yerr=portfolio_summary["revenue_std"],
        fmt="o",
        capsize=4,
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Daily Revenue ($)")
    plt.title("Daily Revenue Mean ± 1 Std Dev by Portfolio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "daily_revenue_variability_by_portfolio.png", dpi=200)
    plt.close()

    print("\nFiles written to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
'''

base = Path("/mnt/data")
(helper_path := base / "market_helpers.py").write_text(textwrap.dedent(helper_code), encoding="utf-8")
(main_path := base / "market_settlement_main.py").write_text(textwrap.dedent(main_code), encoding="utf-8")

print(f"Wrote:\n- {helper_path}\n- {main_path}")
'''