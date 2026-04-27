#!/usr/bin/env bash
set -euo pipefail

# Run both benchmark scripts from the repository root:
#   bash run_benchmarks.sh

: <<'ALIAS_ARGS'
Alias benchmark accepted args:
  --incubator
    Optional incubator name, e.g. github. Omit to run every incubator in
    ref/params.json.
  --gambit-cache-dir
    Directory where Gambit disambiguation CSVs are cached per incubator.
    Optimal default: reports/gambit.
  --refresh-gambit
    Re-run Gambit even when cached incubator CSVs exist. Omit by default to
    reuse cache.
  --output-dir
    Directory for alias_comparison.csv, alias_summary.csv, and alias_status.csv.
    Optimal default: reports/alias_benchmark.

To customize, edit the python command below directly.
ALIAS_ARGS
python3 -m dfc.benchmark.alias \
  --gambit-cache-dir reports/gambit \
  --output-dir reports/alias_benchmark

: <<'BOT_ID_ARGS'
Bot identification benchmark accepted args:
  --incubator
    Optional incubator name, e.g. github. Omit to run every incubator in
    ref/params.json.
  --rabbit-cache-dir
    Directory where RABBIT prediction CSVs are cached per incubator.
    Optimal default: reports/rabbit.
  --refresh-rabbit
    Re-run RABBIT even when cached incubator CSVs exist. Omit by default to
    reuse cache.
  --min-events
    RABBIT min_events argument. Optimal default: 5.
  --min-confidence
    RABBIT min_confidence argument. Optimal default: 1.0.
  --max-queries
    RABBIT max_queries argument. Optimal default: 3.
  --no-wait
    Pass no_wait=True to RABBIT. Omit by default.
  --output-dir
    Directory for bot_id_comparison.csv, bot_id_summary.csv, and
    bot_id_status.csv. Optimal default: reports/bot_id_benchmark.

To customize, edit the python command below directly.
BOT_ID_ARGS
python3 -m dfc.benchmark.bot_id \
  --rabbit-cache-dir reports/rabbit \
  --min-events 5 \
  --min-confidence 1.0 \
  --max-queries 3 \
  --output-dir reports/bot_id_benchmark
