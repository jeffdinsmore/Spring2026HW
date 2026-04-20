from pathlib import Path
import textwrap

helper_code = r'''
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
