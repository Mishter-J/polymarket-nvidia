#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 4 hours ahead.

Reads one case at a time (/input/<id>.in): every trading market of the event,
each with its Yes price at every minute of the 72 hours up to a checkpoint.
Writes /output/<id>.out: for each market and each of the 240 minutes after
the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

Method (tuned on the six practice cases, validated on an earlier 24h window
built from Polymarket's public history):
  * Median: persistence. The last price is the best median forecast these
    markets admit: short-horizon drift and mean reversion toward a 24h level
    both made the practice loss strictly worse, so the median stays put.
  * Spread: per-market, from the last 12 hours of 1-minute moves. Prices
    here are mostly flat with occasional jumps, so the spread is estimated
    from the winsorized (99th percentile) mean absolute 1-minute move,
    converted to a normal sigma (x sqrt(pi/2)), floored at 0.0005 and then
    scaled by 0.7. The house baseline's one-size-fits-all 0.0025 is
    far too wide for the typical market: most of its loss is the width itself.
  * Jump clustering: after a big move (|d| > 0.02) volatility stays high
    for about an hour (measured: mean |d| 5.3x higher in the 60 minutes
    after a jump), so the sigma is boosted by 1 + 0.5*exp(-lag/60min)
    after the most recent big move.
  * Quantiles: normal z-values (-1.644854, -0.67449, 0, 0.67449, 1.644854)
    times sigma times sqrt(horizon), clamped to [0, 1], rounded to 6 decimals.

Practice pinball loss 0.272352 vs the house baseline's 0.363589 (-25%).
Standard library only, no network, deterministic.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Standard normal quantiles at the five scored levels.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

# Tuned constants (see module docstring).
SIGMA_WINDOW_MIN = 12 * 60   # history used for the per-market sigma
SIGMA_SCALE = 0.7            # shrink on the raw robust sigma estimate
SIGMA_FLOOR = 0.0005         # minimum sigma, per sqrt-minute
JUMP_THRESHOLD = 0.02        # a "big move" in one minute
JUMP_BOOST = 0.5             # extra sigma right after a big move
JUMP_TAU_MIN = 60.0          # decay of the post-jump boost, minutes
JUMP_LOOKBACK_MIN = 240      # how far back to look for the last big move


def _round6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def _robust_sigma(prices):
    """Per-minute sigma from the last SIGMA_WINDOW_MIN minutes of moves.

    Winsorized mean absolute 1-minute move, converted to a normal sigma.
    Falls back to the floor when there are too few moves to estimate.
    """
    n = min(len(prices) - 1, SIGMA_WINDOW_MIN)
    if n < 5:
        return SIGMA_FLOOR
    moves = [abs(prices[-1 - i] - prices[-2 - i]) for i in range(n)]
    ordered = sorted(moves)
    cap = ordered[int(len(ordered) * 0.99)]
    clipped = [m if m < cap else cap for m in moves]
    mean_abs = sum(clipped) / len(clipped)
    sigma = mean_abs * 1.2533141373155001  # sqrt(pi/2)
    return max(sigma, SIGMA_FLOOR)


def _jump_boost(prices):
    """Multiplier for sigma after a recent big move (volatility clusters)."""
    n = min(len(prices) - 1, JUMP_LOOKBACK_MIN)
    for lag in range(n):
        if abs(prices[-1 - lag] - prices[-2 - lag]) > JUMP_THRESHOLD:
            return 1.0 + JUMP_BOOST * math.exp(-lag / JUMP_TAU_MIN)
    return 1.0


def forecast_market(prices, horizon):
    """240 rows (or `horizon` rows) of five quantiles for one market."""
    if not prices:
        raise ValueError("market has no prices")
    x = prices[-1]
    # A market with a single price has no moves to learn from: use the floor
    # sigma (persistence median, minimum spread) rather than failing.
    if len(prices) < 2:
        sigma = SIGMA_FLOOR * SIGMA_SCALE
    else:
        sigma = _robust_sigma(prices) * SIGMA_SCALE * _jump_boost(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = [_round6(x + z * spread) for z in Z]
        for i in range(1, 5):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(p.name for p in in_dir.glob("*.in")):
        case_id = name[: -len(".in")]
        with open(in_dir / name, encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        forecasts = [
            {"id": m["id"], "q": forecast_market(m["prices"], horizon)}
            for m in case["markets"]
        ]
        with open(out_dir / f"{case_id}.out", "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
