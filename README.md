# GTA VI Twitch CCU Forecaster

Forecasts, hour by hour, how many people watch **Grand Theft Auto VI** live
on Twitch during each hour of its release day (2026-11-19 UTC), as five
quantiles per hour — an entry for
[`gta6-twitch-ccu-forecast-01`](https://make.fun) ($250 prize pool).

## Method: comparable-launch log-ratio quantiles

The case ships 38 past big game launches with 17 days of hourly Twitch
category averages each. For every comparable, the release day's 24 hourly
prints are extracted and the log-ratio `ln(viewers_h / viewers_1)` is
collected for each hour `h = 2..24`.

- **Reference class:** only launches already live in hour 1 are kept
  (release-day peak at most 20× the hour-1 print). A launch whose main
  unlock lands mid-day UTC has a near-zero hour-1 print and a
  peak/hour-1 ratio in the hundreds; its ratios describe a different shape
  (a dead hour 1, then a vertical ramp). GTA VI unlocks in New Zealand at
  11:00Z on Nov 18, central Europe at 23:00Z, and the UK at 00:00Z, so its
  hour 1 already carries three regions — the live-at-hour-1 class is the
  right reference.
- **Forecast:** per hour, the five scored quantiles are the empirical
  quantiles of the kept launches' log-ratios (winsorised to ±2.5),
  exponentiated against the case's own hour-1 print. The pinball loss is
  scored on the log scale, so forecasting in log space *is* the loss: the
  median traces the typical launch-day curve (high at hour 1, an overnight
  dip, a US-unlock ramp around hours 6–9, an evening peak), and the spread
  is the honest cross-launch dispersion of that curve.
- **Fallback:** with no usable comparable, the house persistence baseline
  (flat at hour 1's print, spread `0.15 · √(h−1)` in log space).

## Validation

Leave-one-out cross-validation over the kit's comparables (148
pseudo-targets, each forecast from the other 37 launches): on launches
already live in hour 1, mean pinball loss **0.3105** vs the persistence
baseline's **0.3595** (13.6% better). Filtering to "Tier: comparable" only
(0.3227) and shrinking the spread toward the median both score worse, so
neither is applied.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Reads every `<id>.in` in the input
directory, writes `<id>.out` per case. Deterministic: no randomness, no wall
clock, no network — two runs are byte-identical.

## License

MIT — see [LICENSE](LICENSE).
