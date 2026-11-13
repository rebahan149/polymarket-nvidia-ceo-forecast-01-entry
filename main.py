"""Quantile fan forecaster for polymarket-nvidia-ceo-forecast-01.

For each market in a case file we emit, for every minute of the horizon,
five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the future Yes price.

How the fan is built:

* The fan is centred on the last observed price. These contracts tend to
  hover near their latest level over a few hours, so the freshest print
  is the best anchor we have.
* Spread comes from a per-market volatility estimate. Looking back over
  the trailing 720 prints, we collect the absolute price changes over
  non-overlapping 30-minute steps and take the upper quartile of those
  jumps. A quartile is robust: a single news spike inside the lookback
  window cannot blow up the whole estimate. The quartile is converted
  to a per-minute sigma by dividing by 1.1503 (its expected multiple of
  sigma for Gaussian steps) and by sqrt(30). Series shorter than 120
  points fall back to a default volatility of 0.0025, and a market whose
  upper quartile never moves is treated as dead flat at 0.0012.
* The raw sigma is clamped into [0.0012, 0.006] and then multiplied by
  0.85; that product is the fan's base width. The 0.006 ceiling is
  deliberately wider than the 0.004 cap a first calibration used: in
  this event the wildest markets are precisely the ones that get
  repriced by a headline (a resignation rumour moves a CEO market more
  than a quiet drift ever will), and capping their volatility too low
  left their fans far too narrow. The cost of the wider tails is paid
  by shrinking every fan a bit harder, hence 0.85 rather than 0.9.
* Width grows with the horizon as h**0.4, slower than the square-root
  law of a free random walk, because a probability bounded in [0, 1]
  mean-reverts instead of wandering off. The five quantile offsets are
  the critical values of Student's t distribution with five degrees of
  freedom, which gives the fan heavier tails than a Gaussian of the
  same width. Every quantile is clamped into [0, 1] and rounded to six
  decimal places.

Calibration: on the six practice cases this recipe scores a mean
pinball loss of about 0.284, against 0.364 for the flat house
reference forecast.

The program uses only the standard library, touches no network, and is
fully deterministic. Cases are read from the directory named by
MAKEFUN_INPUT_DIR (default /input); one output file per case is
written to MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
from pathlib import Path

# Student-t (df = 5) critical values at the five requested quantiles.
T_CRITS = (-2.015048, -0.726687, 0.0, 0.726687, 2.015048)

LOOKBACK = 720          # prints of history feeding the volatility estimate
MIN_POINTS = 120        # below this the fallback volatility is used
FALLBACK_VOL = 0.0025   # volatility for too-short series
FLAT_VOL = 0.0012       # volatility for a series whose quartile never moves
STEP = 30               # minutes between the endpoints of each measured jump
QUARTILE_SCALE = 1.1503  # upper-quartile-to-sigma factor for a 30-min step
VOL_MIN = 0.0012
VOL_MAX = 0.006
SHRINK = 0.85           # global narrowing applied after the clamp
GROWTH = 0.4            # horizon exponent of the fan width (sub-diffusive:
                        # the fan widens more slowly than sqrt(h) would)


def estimate_vol(prices):
    """Per-minute volatility implied by the tail of the price history."""
    window = prices[-LOOKBACK:]
    if len(window) < MIN_POINTS:
        return FALLBACK_VOL
    jumps = [abs(window[i] - window[i - STEP])
             for i in range(STEP, len(window), STEP)]
    if not jumps:
        return FLAT_VOL
    jumps.sort()
    upper = jumps[(3 * (len(jumps) - 1)) // 4]
    if upper == 0:
        return FLAT_VOL
    return upper / QUARTILE_SCALE / math.sqrt(STEP)


def fan_for_market(prices, horizon):
    """Build the horizon x 5 quantile table for one market."""
    series = prices if prices else [0.5]
    anchor = series[-1]
    vol = min(VOL_MAX, max(VOL_MIN, estimate_vol(series)))
    base_width = vol * SHRINK
    table = []
    for h in range(1, horizon + 1):
        width = base_width * h ** GROWTH
        row = [round(min(1.0, max(0.0, anchor + c * width)), 6)
               for c in T_CRITS]
        table.append(row)
    return table


def forecast_case(case):
    """Turn one parsed case document into its forecast document."""
    horizon = int(case.get("horizon_minutes", 240))
    forecasts = []
    for market in case["markets"]:
        forecasts.append({
            "id": market["id"],
            "q": fan_for_market(market["prices"], horizon),
        })
    return {"forecasts": forecasts}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_file in sorted(in_dir.glob("*.in")):
        with open(case_file, encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = out_dir / (case_file.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)


if __name__ == "__main__":
    main()
