"""Quantile fan forecasts for the polymarket-nvidia-ceo-forecast-01 task.

Each case file holds several markets with a price history. For every
market this program projects the Yes price over the coming minutes and
reports five quantiles per minute: 0.05, 0.25, 0.5, 0.75 and 0.95.

Forecast recipe, calibrated against the six practice cases (mean pinball
loss 0.284, while the flat house reference scores 0.364):

* The centre of the fan is pinned to the most recent observed price:
  within a few hours these contracts usually stay close to that level.
* The half-width grows with the horizon raised to 0.4, which is slower
  than square-root growth; bounded, mean-reverting prices do not fan
  out like a free random walk.
* The tail shape is a Student t distribution with five degrees of
  freedom, wide enough for the sudden jumps these series produce.
* Volatility is measured market by market, using a robust quartile of
  the absolute half-hour price changes across the trailing twelve
  hours, so one news-driven spike in the lookback cannot inflate the
  whole fan; a market whose series history is too short or completely
  flat receives a fixed default volatility instead.
* The volatility ceiling sits higher than a first calibration suggested
  (0.006, not 0.004): in a field where one market can be repriced by a
  resignation headline, the wildest series are exactly the ones whose
  fans the ceiling would otherwise clip too tight. The overall fan is
  then shrunk a little harder (0.85) to pay for the wider tails.

Only the Python standard library is used and the output is fully
deterministic. Input cases are read from the directory named by the
environment variable MAKEFUN_INPUT_DIR (falling back to /input) and one
result file per case is written to MAKEFUN_OUTPUT_DIR (or /output).
"""

import json, math, os
from pathlib import Path

# Critical values of the t distribution with five degrees of
# freedom, taken at the five quantile levels this task asks for, in order.
T_CRITICALS = (-2.015048, -0.726687, 0.0, 0.726687, 2.015048)

SHORT_HISTORY_VOL = 0.0025  # fallback volatility for series under 120 points
FLAT_MARKET_VOL = 0.0012    # floor volatility for a series that never moved
VOL_FLOOR = 0.0012
VOL_CEILING = 0.006
FAN_SHRINK = 0.85           # empirical calibration factor for the fan width
GROWTH_POWER = 0.4          # sub-diffusive growth of the fan with horizon
HISTORY_SPAN = 720          # minutes of trailing history used for volatility
JUMP_LAG = 30               # lag, in minutes, between compared price points
QUARTILE_DIVISOR = 1.1503   # scales the upper quartile of moves to a sigma


def per_market_vol(series):
    """Robust per-minute volatility implied by a price history."""
    window = series[-HISTORY_SPAN:]
    if len(window) < 120:
        return SHORT_HISTORY_VOL
    jumps = []
    for start in range(JUMP_LAG, len(window), JUMP_LAG):
        jumps.append(abs(window[start] - window[start - JUMP_LAG]))
    jumps.sort()
    if len(jumps) == 0:
        return SHORT_HISTORY_VOL
    upper = jumps[(3 * (len(jumps) - 1)) // 4]
    if not upper > 0.0:
        return FLAT_MARKET_VOL
    return upper / QUARTILE_DIVISOR / math.sqrt(JUMP_LAG)


def clamp_price(value):
    """Keep a probability forecast inside its feasible interval."""
    return min(1.0, max(0.0, value))


def build_row(anchor, width):
    """One row of the fan: the anchor shifted by each critical value."""
    return [round(clamp_price(anchor + crit * width), 6) for crit in T_CRITICALS]


def project_market(series, minutes):
    """Project one market: a fan of quantile rows, one per minute."""
    anchor = series[-1]
    vol = per_market_vol(series)
    width_base = min(VOL_CEILING, max(VOL_FLOOR, vol)) * FAN_SHRINK
    table = []
    for step in range(1, minutes + 1):
        table.append(build_row(anchor, width_base * (step ** GROWTH_POWER)))
    return table


def solve_case(payload):
    """Build the result document for one parsed case file."""
    horizon = int(payload.get("horizon_minutes", 240))
    forecasts = []
    for entry in payload["markets"]:
        history = entry["prices"] if entry["prices"] else [0.5]
        forecasts.append({"id": entry["id"], "q": project_market(history, horizon)})
    return {"forecasts": forecasts}


def main():
    source_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    target_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    target_dir.mkdir(parents=True, exist_ok=True)
    for case_path in sorted(source_dir.glob("*.in")):
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        result_path = target_dir.joinpath(case_path.stem + ".out")
        result_path.write_text(json.dumps(solve_case(payload)), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
