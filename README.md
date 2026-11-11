# Roblox Up-and-Coming chart forecaster

Entry for `roblox-up-and-coming-forecast-01` on make.fun, developed by `developertradeat2026-tech`.

## Model

The program predicts a separate marginal distribution for each of the chart's top 10 places, averaged over every hour of the requested 72-hour forecast window. It fits three mechanisms solely to the supplied history:

- Origin-rank retention, allowing a different persistence curve at each current top-10 rank.
- A stationary rank-diffusion kernel whose width grows with elapsed hours, capturing movement between nearby places and inward drift from outside the top 10.
- Per-rank-band membership curves, distinguishing established games near the top from deeper games that only occasionally enter the top 10.

A separate residual pool spreads mass across games that never appeared in history, rather than forcing the model to over-concentrate on the incumbent chart. The reported `top10` uses a greedy assignment that maximizes the probability-weighted, position-aware agreement tie-break.

## Requirements and runtime

- Python 3 (tested with Python 3.12; uses no nonstandard features).
- Standard library only; no installation and no network calls.
- `/bot` may be read-only. All output is written to `/output`.
- Two runs on the practice history produced byte-identical output.
- Measured local execution: approximately 0.2 seconds and 30 MB RSS, comfortably inside the 300-second / 1-CPU / 1024-MB limits.

Run locally with an `input/` directory containing `task.json` and `history.ndjson`:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

In the grading container it is invoked as:

```sh
python3 /bot/main.py
```

## Practice result

The kit exposes only the pre-history, not the 10 scored practice reads. Those reads were reconstructed locally from the same public MIT-licensed `Kenny-Dhub/roblox-snapshots` archive specified by `competition.yaml`. The scoring code follows the public per-place logarithmic scoring rule independently.

- Reconstructed persistence baseline: **-43.512039268**, matching the published **-43.512039**.
- This program: **-36.619304522** on the same 10 reads.
- Improvement over the persistence baseline: **+6.892735 nats**.

The score takes the mean over the 10 scored reads of the sum over the top-10 places of `ln max(P, 0.0001)`, matching the competition's baseline exactly.

## Development verification

- Walk-forward tests fit each prefix independently and score at its actual horizon.
- Evaluating one averaged 72-hour forecast on each prefix across 185 walk-forward evaluations gave a **+9.33 nats/read** mean improvement over baseline, with **93.5%** better than baseline.
- The final output passes independent JSON-schema, probability-range, uniqueness, weight-sum, and `top10` checks.
- A stress run with 120 hourly reads completed without hitting the wall limit.
- Two fresh runs on the same 34-read practice history were byte-identical.

The kit's `grade-local` command refuses this `program/v1` competition because its scorer/reference data are not published in the kit and it grades on the server. The reconstruction above verifies the practice result from the named public source rather than claiming a server-side harness run.

## License

MIT. See `LICENSE`.
