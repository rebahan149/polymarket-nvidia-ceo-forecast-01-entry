"""polymarket-nvidia-ceo-forecast-01 entry: minute-by-minute quantile forecasts.

For every market in a case file, forecast the Yes price at each of the next
240 minutes as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

Model, tuned on the six practice cases (pinball loss 0.289 vs the house
baseline's 0.364): the median stays at the checkpoint price; the spread is a
Student-t (df=5) fan whose per-market sigma is estimated robustly from the
75th percentile of absolute 30-minute moves over the last 12 hours, so a
single news spike in the lookback does not blow the fan wide open. The fan
widens like h**0.4 (slower than a random walk: these are bounded,
mean-reverting prices) and is clamped to [0, 1] like the prices themselves.

Python 3.13, standard library only, deterministic.
Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR
(default /output).
"""

import json
import math
import os
from pathlib import Path

Z = [-2.015048, -0.726687, 0.0, 0.726687, 2.015048]  # Student-t df=5 quantiles
FALLBACK_SIGMA = 0.0025  # the house baseline's global sigma
ZERO_SIGMA = 0.0012      # a market that did not move still gets a narrow fan
SIGMA_LO = 0.0012
SIGMA_HI = 0.004
SCALE = 0.9
BETA = 0.4
LOOKBACK = 720   # minutes of history used for the vol estimate (12 h)
STEP = 30        # move horizon (minutes) the vol is measured on


def sigma_of(prices: list[float]) -> float:
    """Per-sqrt-minute vol from the 75th percentile of |30-min moves|."""
    p = prices[-LOOKBACK:]
    if len(p) < 120:
        return FALLBACK_SIGMA
    moves = sorted(abs(p[i] - p[i - STEP]) for i in range(STEP, len(p), STEP))
    if not moves:
        return FALLBACK_SIGMA
    q75 = moves[int(0.75 * (len(moves) - 1))]
    if q75 <= 0:
        return ZERO_SIGMA
    return q75 / 1.1503 / math.sqrt(STEP)


def forecast_market(prices: list[float], horizon: int) -> list[list[float]]:
    x = prices[-1]
    s = min(SIGMA_HI, max(SIGMA_LO, sigma_of(prices))) * SCALE
    rows = []
    for h in range(1, horizon + 1):
        w = s * (h ** BETA)
        rows.append([round(min(1.0, max(0.0, x + z * w)), 6) for z in Z])
    return rows


def run_case(case: dict) -> dict:
    horizon = int(case.get("horizon_minutes", 240))
    return {
        "forecasts": [
            {"id": m["id"], "q": forecast_market(m["prices"] or [0.5], horizon)}
            for m in case["markets"]
        ]
    }


def main() -> None:
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = run_case(case)
        out_path = out_dir / (path.stem + ".out")
        out_path.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
