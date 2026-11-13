# polymarket-nvidia-ceo-forecast-01 — quantile fan forecast

Forecasts Polymarket's "Next CEO of NVIDIA?" Yes prices, one minute at a
time, four hours ahead: for every market in a case, 240 rows of five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95). Python 3.13, standard library
only, deterministic — the same case always produces the same file.

The model, tuned on the six practice cases against the real recorded
prices (pinball loss 0.289, where the house baseline scores 0.364):

- The median stays at the checkpoint price — over four hours these
  markets mostly go nowhere.
- The spread is a Student-t (df=5) fan: fat enough tails for the jumps
  these prices make, without paying for width in the quiet minutes.
- Each market's sigma is its own, estimated robustly from the 75th
  percentile of its absolute 30-minute moves over the last 12 hours, so
  one news spike in the lookback does not blow the fan wide open, and a
  dead market still gets a narrow-but-honest fan.
- The fan widens like h**0.4, slower than a random walk: these are
  bounded, mean-reverting prices, and the practice cases punish
  square-root widening.

Reads the case files in `$MAKEFUN_INPUT_DIR` (default `/input`) and
writes one `.out` file per case to `$MAKEFUN_OUTPUT_DIR`
(default `/output`).
