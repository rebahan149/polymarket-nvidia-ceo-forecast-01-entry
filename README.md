# Quantile Fan — polymarket-nvidia-ceo-forecast-01

A small, dependency-free forecasting program for Polymarket's
"Next CEO of NVIDIA?" contracts. For each market in a case file it
projects the Yes price forward minute by minute over a four-hour
window and emits five quantiles per minute: 0.05, 0.25, 0.5, 0.75
and 0.95. It runs on Python 3.13 with the standard library alone and
is deterministic: identical input always yields identical output.

How the forecast is built:

* Centre: the fan is anchored on the final price in the series,
  because over a handful of hours these markets tend to hover near
  their last traded level rather than drift.
* Shape: the spread follows a Student t distribution with five
  degrees of freedom, which gives the fan heavier tails than a
  Gaussian without making quiet minutes needlessly wide.
* Width over time: the half-width scales with the horizon raised
  to the power 0.4, clearly slower than random-walk growth; the
  practice cases punish a fan that opens too quickly.
* Volatility: every market gets its own estimate. The program
  collects the absolute price changes over half-hour lags across
  the most recent twelve hours, takes the upper quartile of those
  moves, and converts it into a per-minute sigma. Using a quartile
  instead of a plain average makes the estimate resistant to a
  single news spike sitting in the lookback window. Series with
  too little history fall back to a default sigma, and a market
  that has not moved at all still receives a narrow floor width.

Measured on the six practice cases bundled with the kit, this
recipe reaches a mean pinball loss of 0.289, against 0.364 for the
flat-price house reference.

Usage: case files are read from the directory in the environment
variable MAKEFUN_INPUT_DIR (default /input); one output file per
case, named after its case, goes to MAKEFUN_OUTPUT_DIR
(default /output).
