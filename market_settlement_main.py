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
    run_full_8day_monte_carlo,
    #run_monte_carlo,
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
OUTPUT_DIR = Path("msg_outputs2")


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
    results = run_full_8day_monte_carlo(
        df_units=df_units,
        n_sims=N_SIMS,
        random_seed=42,
        bid_adder=BID_ADDER,
    )

    days_1_4 = results["days_1_4_unconstrained"]
    days_5_6 = results["days_5_6_constrained_2500"]
    days_7_8 = results["days_7_8_constrained_2000"]
    combined_8day = results["combined_8day_total"]

    portfolio_summary = combined_8day["portfolio_summary"].copy()
    #hourly_price_summary = results["hourly_price_summary"].copy()
    #all_daily = results["all_daily_results"].copy()

    # Save CSV outputs.
    # Days 1-4 outputs
    days_1_4["portfolio_summary"].to_csv(
        OUTPUT_DIR / "days_1_4_portfolio_summary.csv", index=False
    )
    days_1_4["hourly_price_summary"].to_csv(
        OUTPUT_DIR / "days_1_4_hourly_price_summary.csv", index=False
    )
    days_1_4["all_daily_results"].to_csv(
        OUTPUT_DIR / "days_1_4_all_daily_results.csv", index=False
    )
    days_1_4["all_dispatch_results"].to_csv(
        OUTPUT_DIR / "days_1_4_dispatch_detailed.csv", index=False
    )

    # Days 5-6 outputs
    days_5_6["portfolio_summary"].to_csv(
        OUTPUT_DIR / "days_5_6_portfolio_summary.csv", index=False
    )
    days_5_6["hourly_price_summary"].to_csv(
        OUTPUT_DIR / "days_5_6_hourly_price_summary.csv", index=False
    )
    days_5_6["all_daily_results"].to_csv(
        OUTPUT_DIR / "days_5_6_all_daily_results.csv", index=False
    )
    days_5_6["all_dispatch_results"].to_csv(
        OUTPUT_DIR / "days_5_6_dispatch_detailed.csv", index=False
    )

    # Days 7-8 outputs
    days_7_8["portfolio_summary"].to_csv(
        OUTPUT_DIR / "days_7_8_portfolio_summary.csv", index=False
    )
    days_7_8["hourly_price_summary"].to_csv(
        OUTPUT_DIR / "days_7_8_hourly_price_summary.csv", index=False
    )
    days_7_8["all_daily_results"].to_csv(
        OUTPUT_DIR / "days_7_8_all_daily_results.csv", index=False
    )
    days_7_8["all_dispatch_results"].to_csv(
        OUTPUT_DIR / "days_7_8_dispatch_detailed.csv", index=False
    )

    # Combined 8-day output
    portfolio_summary.to_csv(
        OUTPUT_DIR / "combined_8day_portfolio_summary.csv", index=False
    )

    print("\n" + "=" * 72)
    print("COMBINED 8-DAY RESULTS")
    print("=" * 72)
    print(portfolio_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("DAYS 1-4 HOURLY PRICE SUMMARY")
    print("=" * 72)
    print(days_1_4["hourly_price_summary"].to_string(index=False))

    print("\n" + "=" * 72)
    print("DAYS 5-6 HOURLY PRICE SUMMARY")
    print("=" * 72)
    print(days_5_6["hourly_price_summary"].to_string(index=False))

    print("\n" + "=" * 72)
    print("DAYS 7-8 HOURLY PRICE SUMMARY")
    print("=" * 72)
    print(days_7_8["hourly_price_summary"].to_string(index=False))

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

    print("\nFiles written to:")
    print(OUTPUT_DIR.resolve())

    # --------------------------------------------------------
    # Step 5: Make plots for the 3 market phases + combined
    # --------------------------------------------------------

    # Plot 1: Days 1-4 expected 4-day profit by portfolio
    plot_days_1_4 = days_1_4["portfolio_summary"].copy()
    plot_days_1_4["expected_4day_profit"] = 4.0 * plot_days_1_4["expected_daily_profit"]

    plt.figure(figsize=(10, 6))
    plt.bar(plot_days_1_4["utility_name"], plot_days_1_4["expected_4day_profit"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Expected Profit ($)")
    plt.title("Days 1-4 Expected Profit by Portfolio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "days_1_4_expected_profit_by_portfolio.png", dpi=200)
    plt.close()


    # Plot 2: Days 5-6 expected 2-day profit by portfolio
    plot_days_5_6 = days_5_6["portfolio_summary"].copy()
    plot_days_5_6["expected_2day_profit"] = 2.0 * plot_days_5_6["expected_daily_profit"]

    plt.figure(figsize=(10, 6))
    plt.bar(plot_days_5_6["utility_name"], plot_days_5_6["expected_2day_profit"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Expected Profit ($)")
    plt.title("Days 5-6 Expected Profit by Portfolio (2500 MW Transfer Limit)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "days_5_6_expected_profit_by_portfolio.png", dpi=200)
    plt.close()


    # Plot 3: Days 7-8 expected 2-day profit by portfolio
    plot_days_7_8 = days_7_8["portfolio_summary"].copy()
    plot_days_7_8["expected_2day_profit"] = 2.0 * plot_days_7_8["expected_daily_profit"]

    plt.figure(figsize=(10, 6))
    plt.bar(plot_days_7_8["utility_name"], plot_days_7_8["expected_2day_profit"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Expected Profit ($)")
    plt.title("Days 7-8 Expected Profit by Portfolio (2000 MW Transfer Limit)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "days_7_8_expected_profit_by_portfolio.png", dpi=200)
    plt.close()


    # Plot 4: Combined 8-day expected profit by portfolio
    plt.figure(figsize=(10, 6))
    plt.bar(portfolio_summary["utility_name"], portfolio_summary["expected_8day_profit"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Expected Profit ($)")
    plt.title("Combined 8-Day Expected Profit by Portfolio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combined_8day_expected_profit_by_portfolio.png", dpi=200)
    plt.close()

    # Plot 5: Days 1-4 expected market clearing price by hour
    price_days_1_4 = days_1_4["hourly_price_summary"].copy()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        price_days_1_4["hour"],
        price_days_1_4["expected_price"],
        yerr=price_days_1_4["price_std"],
        marker="o",
        linestyle="-",
        capsize=4,
    )
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("Hour")
    plt.ylabel("Market Clearing Price ($/MWh)")
    plt.title("Days 1-4 Expected Market Clearing Price by Hour")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "days_1_4_market_clearing_price_by_hour.png", dpi=200)
    plt.close()


    # Plot 6: Days 5-6 expected West and East market clearing price by hour
    price_days_5_6 = days_5_6["hourly_price_summary"].copy()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        price_days_5_6["hour"],
        price_days_5_6["expected_west_price"],
        yerr=price_days_5_6["west_price_std"],
        marker="o",
        linestyle="-",
        capsize=4,
        label="West",
    )
    plt.errorbar(
        price_days_5_6["hour"],
        price_days_5_6["expected_east_price"],
        yerr=price_days_5_6["east_price_std"],
        marker="s",
        linestyle="-",
        capsize=4,
        label="East",
    )
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("Hour")
    plt.ylabel("Market Clearing Price ($/MWh)")
    plt.title("Days 5-6 Expected Market Clearing Price by Hour")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "days_5_6_market_clearing_price_by_hour.png", dpi=200)
    plt.close()


    # Plot 7: Days 7-8 expected West and East market clearing price by hour
    price_days_7_8 = days_7_8["hourly_price_summary"].copy()

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        price_days_7_8["hour"],
        price_days_7_8["expected_west_price"],
        yerr=price_days_7_8["west_price_std"],
        marker="o",
        linestyle="-",
        capsize=4,
        label="West",
    )
    plt.errorbar(
        price_days_7_8["hour"],
        price_days_7_8["expected_east_price"],
        yerr=price_days_7_8["east_price_std"],
        marker="s",
        linestyle="-",
        capsize=4,
        label="East",
    )
    plt.xticks([1, 2, 3, 4])
    plt.xlabel("Hour")
    plt.ylabel("Market Clearing Price ($/MWh)")
    plt.title("Days 7-8 Expected Market Clearing Price by Hour")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "days_7_8_market_clearing_price_by_hour.png", dpi=200)
    plt.close()

    """
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
    """

    


if __name__ == "__main__":
    main()
'''

base = Path("/mnt/data")
(helper_path := base / "market_helpers.py").write_text(textwrap.dedent(helper_code), encoding="utf-8")
(main_path := base / "market_settlement_main.py").write_text(textwrap.dedent(main_code), encoding="utf-8")

print(f"Wrote:\n- {helper_path}\n- {main_path}")
'''