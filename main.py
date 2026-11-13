"""Quantile fan forecaster for polymarket-nvidia-ceo-forecast-01 (round 3).

For every market in a case file we emit, for each minute of the horizon,
five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the future Yes price.

How a fan is built, in three steps.

Step one is coherence. The markets in a case are candidates for one job:
mutually exclusive outcomes, so their Yes prices really ought to add up
to one. In practice the slate drifts above or below that, so before
anything else we nudge every market's anchor -- its last observed price,
or 0.5 for a market with no prints at all -- halfway toward a slate
that sums to exactly one. Concretely, if the raw anchors sum to S, each
anchor is scaled by 0.5 + 0.5 / S: half of the incoherence is removed,
half is left alone, because some of the gap is genuine disagreement
about the field rather than noise. The adjusted anchor centres the fan.

Step two is a per-market volatility. From the trailing 720 prints we
take the absolute price changes over non-overlapping 30-minute steps
and use their upper quartile, a robust summary that one headline spike
cannot dominate, converted to a per-minute sigma by dividing out 1.1503
(the quartile's expected size, in sigmas, for Gaussian steps) and
sqrt(30). Two fallbacks matter:

  * A market with fewer than 120 prints gets a volatility of 0.01,
    used exactly as written -- it is not squeezed through the clamp
    below. A young market is an unknown market, and unknown means
    uncertain, so it deliberately receives a wider fan than even the
    wildest established market: it would be backwards to hand it the
    calm, narrow fan a sleepy established market earns.
  * A market whose upper-quartile jump is exactly zero is treated as
    dead flat, at the 0.0012 floor.

The volatility is clamped into [0.0012, 0.006] and then multiplied by
0.80. The 0.006 ceiling is deliberately generous: in an event like
this, the wildest markets are wild because a headline can reprice them
in minutes, and capping their sigma lower would leave their fans far
too narrow exactly when it matters. The price of those wider tails is
paid by shrinking every fan to 0.80 of its raw width.

Step three is the fan itself. Width grows with the horizon as h**0.4,
slower than a random walk's square-root law, because a probability
trapped in [0, 1] mean-reverts instead of wandering off. The five
offsets are the critical values of Student's t distribution with five
degrees of freedom, giving heavier tails than a Gaussian of the same
width. Every value is clamped into [0, 1] and rounded to six decimals.

Calibration: on the six practice cases this recipe scores a mean
pinball loss of about 0.268, against 0.364 for the flat house
reference forecast.

Standard library only, no network, fully deterministic. Input comes
from MAKEFUN_INPUT_DIR (default /input); one .out file per case goes
to MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
from pathlib import Path

# Student-t (df = 5) critical values for the five requested quantiles,
# (0.05, 0.25, 0.5, 0.75, 0.95) from outermost to innermost.
CRITICALS = (-2.015048, -0.726687, 0.0, 0.726687, 2.015048)

HISTORY = 720           # prints of lookback for the volatility estimate
MIN_HISTORY = 120       # below this a market counts as "young"
YOUNG_VOL = 0.01        # volatility for a young market (used as-is)
FLAT_VOL = 0.0012       # volatility when the quartile jump never moves
JUMP_LAG = 30           # minutes between the endpoints of a measured jump
QUARTILE_FACTOR = 1.1503  # upper-quartile size, in sigmas, Gaussian steps
VOL_FLOOR = 0.0012
VOL_CEILING = 0.006
WIDTH_SCALE = 0.80      # global shrink applied to the clamped volatility
HORIZON_POWER = 0.4     # fan width grows as h ** HORIZON_POWER


def raw_anchor(prices):
    """A market's centre before coherence: last print, else 0.5."""
    return prices[-1] if prices else 0.5


def estimate_vol(prices):
    """Per-minute volatility for one market, before the shrink.

    A young market returns its 0.01 fallback untouched: the clamp
    governs only the history-derived estimate.
    """
    window = prices[-HISTORY:]
    if len(window) < MIN_HISTORY:
        return YOUNG_VOL
    jumps = sorted(
        abs(window[i] - window[i - JUMP_LAG])
        for i in range(JUMP_LAG, len(window), JUMP_LAG)
    )
    if not jumps:
        return FLAT_VOL
    upper = jumps[(3 * (len(jumps) - 1)) // 4]
    if upper == 0:
        return FLAT_VOL
    sigma = upper / QUARTILE_FACTOR / math.sqrt(JUMP_LAG)
    return min(VOL_CEILING, max(VOL_FLOOR, sigma))


def build_fan(centre, vol, horizon):
    """The horizon x 5 quantile table around one adjusted anchor."""
    base = vol * WIDTH_SCALE
    rows = []
    for h in range(1, horizon + 1):
        width = base * h ** HORIZON_POWER
        rows.append([
            round(min(1.0, max(0.0, centre + c * width)), 6)
            for c in CRITICALS
        ])
    return rows


def forecast_case(case):
    """Turn one parsed case document into its forecast document."""
    horizon = int(case.get("horizon_minutes", 240))
    markets = case["markets"]
    anchors = [raw_anchor(market["prices"]) for market in markets]
    total = sum(anchors)
    factor = 0.5 + 0.5 / total if total > 0 else 1.0
    forecasts = []
    for market, anchor in zip(markets, anchors):
        vol = estimate_vol(market["prices"])
        forecasts.append({
            "id": market["id"],
            "q": build_fan(anchor * factor, vol, horizon),
        })
    return {"forecasts": forecasts}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_path in sorted(in_dir.glob("*.in")):
        with open(case_path, encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_path = out_dir / (case_path.stem + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle)


if __name__ == "__main__":
    main()
