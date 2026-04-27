"""
Benchmark DFC rawdata bot labels against RABBIT predictions.

1. Load RawData with its default versions; the data must already contain DFC's
   ``is_bot`` field.
2. Run RABBIT on the contributors present in that rawdata, using a GitHub API
   key loaded from dotenv.
3. Join RABBIT predictions onto the same rows as ``rabbit_is_bot``.
4. Compare the two labels.
5. Repeat for every incubator listed in ``ref/params.json``.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from dfc.abstractions.rawdata import RawData
from dfc.utils import check_path, load_params, log


RABBIT_BOT_TYPE = "bot"
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
RABBIT_CONTRIBUTOR_FIELDS = ("contributor", "login", "username", "CONTRIBUTOR")
RABBIT_TYPE_FIELDS = ("type", "TYPE", "rabbit_type", "prediction", "label")
RABBIT_CONFIDENCE_FIELDS = (
    "confidence",
    "CONFIDENCE",
    "rabbit_confidence",
    "score",
    "probability",
)
RABBIT_FEATURE_FIELDS = ("features", "FEATURES", "rabbit_features")


def _first_existing(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate that appears in ``columns``."""

    lower_lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def load_rawdata(incubator: str) -> dict[str, pd.DataFrame]:
    """Load local rawdata through RawData's default-version handling."""

    return RawData(incubator=incubator).data


def load_dotenv_api_key() -> str:
    """Load GITHUB_API_KEY from the repo-root dotenv file."""

    load_dotenv(DEFAULT_ENV_PATH)
    api_key = os.getenv("GITHUB_API_KEY")
    if not api_key:
        raise ValueError(f"Could not find GITHUB_API_KEY in {DEFAULT_ENV_PATH}")
    return api_key


def rawdata_contributors(
    data_lookup: dict[str, pd.DataFrame],
    incubator: str,
    params: dict[str, Any],
) -> list[str]:
    """Extract unique contributors from the same rawdata passed to comparison."""

    author_field = params["author-source-field"][incubator]
    contributors: set[str] = set()

    for activity_type, df in data_lookup.items():
        if author_field not in df.columns:
            raise ValueError(
                f"{activity_type} rawdata for {incubator} is missing "
                f"{author_field!r}"
            )

        contributors.update(
            str(value)
            for value in df[author_field].dropna().unique()
            if str(value).strip() and str(value).lower() != "none"
        )

    return sorted(contributors)


def _normalize_rabbit_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a local RABBIT result table."""

    columns = set(df.columns)
    contributor_col = _first_existing(columns, RABBIT_CONTRIBUTOR_FIELDS)
    type_col = _first_existing(columns, RABBIT_TYPE_FIELDS)
    confidence_col = _first_existing(columns, RABBIT_CONFIDENCE_FIELDS)
    features_col = _first_existing(columns, RABBIT_FEATURE_FIELDS)

    if contributor_col is None or type_col is None:
        raise ValueError(
            "RABBIT results must include contributor and type columns. "
            f"Found columns: {list(df.columns)}"
        )

    normalized = pd.DataFrame(
        {
            "contributor": df[contributor_col].astype(str),
            "rabbit_type": df[type_col].astype(str),
        }
    )
    if confidence_col is not None:
        normalized["rabbit_confidence"] = pd.to_numeric(
            df[confidence_col], errors="coerce"
        )
    else:
        normalized["rabbit_confidence"] = pd.NA

    if features_col is not None:
        normalized["rabbit_features"] = df[features_col]

    normalized["rabbit_is_bot"] = (
        normalized["rabbit_type"].str.lower() == RABBIT_BOT_TYPE
    ).astype(int)
    return normalized.drop_duplicates(subset=["contributor"], keep="last")


def _read_rabbit_csv(path: str | Path) -> pd.DataFrame:
    """Read a RABBIT CSV export."""

    df = pd.read_csv(path)
    return _normalize_rabbit_predictions(df)


def _load_rabbit_runner():
    """Load RABBIT's Python runner from the installed package."""

    for module_name in ("rabbit", "rabbit_ng"):
        try:
            module = import_module(module_name)
        except ImportError:
            continue

        runner = getattr(module, "run_rabbit", None)
        if runner is not None:
            return runner

    raise ImportError(
        "Could not import RABBIT's Python API. Install the RABBIT package that "
        "exports `run_rabbit`, then retry."
    )


def run_rabbit_predictions(
    contributors: list[str],
    github_api_key: str,
    incubator: str,
    min_events: int = 5,
    min_confidence: float = 1.0,
    max_queries: int = 3,
    no_wait: bool = False,
) -> pd.DataFrame:
    """Run RABBIT for contributors and return normalized predictions."""

    contributors = sorted({c for c in contributors if c and c.lower() != "none"})
    if not contributors:
        return pd.DataFrame(
            columns=["contributor", "rabbit_type", "rabbit_confidence", "rabbit_is_bot"]
        )

    rabbit_run = _load_rabbit_runner()
    rows: list[dict[str, Any]] = []
    results = rabbit_run(
        contributors=contributors,
        api_key=github_api_key,
        min_events=min_events,
        min_confidence=min_confidence,
        max_queries=max_queries,
        no_wait=no_wait,
    )
    for result in results:
        user_type = getattr(result, "user_type", getattr(result, "type", None))
        rows.append(
            {
                "contributor": result.contributor,
                "rabbit_type": user_type,
                "rabbit_confidence": result.confidence,
                "rabbit_features": json.dumps(
                    getattr(result, "features", None),
                    default=str,
                ),
            }
        )
    result_df = pd.DataFrame.from_records(rows)
    if result_df.empty:
        return pd.DataFrame(
            columns=["contributor", "rabbit_type", "rabbit_confidence", "rabbit_is_bot"]
        )

    tqdm.pandas(desc=f"{incubator}: RABBIT contributors")
    result_df["contributor"] = result_df["contributor"].progress_apply(str)
    return _normalize_rabbit_predictions(result_df)


def attach_rabbit_predictions(
    data_lookup: dict[str, pd.DataFrame],
    incubator: str,
    rabbit_predictions: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Join RABBIT labels onto each rawdata dataframe."""

    author_field = params["author-source-field"][incubator]
    enriched: dict[str, pd.DataFrame] = {}

    for activity_type, df in data_lookup.items():
        required = {author_field, "is_bot"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{activity_type} rawdata for {incubator} is missing columns: "
                f"{sorted(missing)}. Load a rawdata version with bot "
                "preprocessing already applied."
            )

        out = df.copy()
        out["is_bot"] = out["is_bot"].fillna(0).astype(int)
        tqdm.pandas(desc=f"{incubator}: attach {activity_type}")
        out["_rabbit_join_contributor"] = out[author_field].progress_apply(str)
        out = out.merge(
            rabbit_predictions,
            left_on="_rabbit_join_contributor",
            right_on="contributor",
            how="left",
        )
        out.drop(columns=["_rabbit_join_contributor", "contributor"], inplace=True)
        out["rabbit_type"] = out["rabbit_type"].fillna("Missing")
        out["rabbit_is_bot"] = out["rabbit_is_bot"].fillna(0).astype(int)
        enriched[activity_type] = out

    return enriched


def rawdata_to_comparison_rows(
    data_lookup: dict[str, pd.DataFrame],
    incubator: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Aggregate enriched rawdata into comparison rows."""

    author_field = params["author-source-field"][incubator]
    records: list[dict[str, Any]] = []

    for activity_type, df in data_lookup.items():
        required = {
            "project_name",
            author_field,
            "is_bot",
            "rabbit_is_bot",
            "rabbit_type",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{activity_type} enriched rawdata for {incubator} is missing "
                f"columns: {sorted(missing)}"
            )

        project_activity = df.groupby("project_name", observed=True).size()
        grouped = (
            df.groupby(["project_name", author_field], observed=True)
            .agg(
                activity_count=(author_field, "size"),
                dfc_is_bot=("is_bot", "max"),
                rabbit_is_bot=("rabbit_is_bot", "max"),
                rabbit_type=("rabbit_type", "last"),
                rabbit_confidence=("rabbit_confidence", "last"),
            )
            .reset_index()
        )

        def build_record(row: pd.Series) -> dict[str, Any]:
            project_name = row["project_name"]
            contributor = str(row[author_field])
            activity_count = int(row["activity_count"])
            total_activity = int(project_activity.loc[project_name])
            return {
                "incubator": incubator,
                "project_name": project_name,
                "activity_type": activity_type,
                "contributor": contributor,
                "activity_count": activity_count,
                "project_activity_count": total_activity,
                "activity_proportion": (
                    activity_count / total_activity if total_activity else 0.0
                ),
                "dfc_is_bot": int(row["dfc_is_bot"]),
                "rabbit_is_bot": int(row["rabbit_is_bot"]),
                "rabbit_type": row["rabbit_type"],
                "rabbit_confidence": row["rabbit_confidence"],
            }

        tqdm.pandas(desc=f"{incubator}: compare {activity_type}")
        records.extend(grouped.progress_apply(build_record, axis=1).tolist())

    comparison = pd.DataFrame.from_records(records)
    if comparison.empty:
        comparison["agreement"] = pd.Series(dtype="int64")
    else:
        comparison["agreement"] = (
            comparison["dfc_is_bot"] == comparison["rabbit_is_bot"]
        ).astype(int)
    return comparison


def summarize_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize DFC-vs-RABBIT agreement."""

    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "incubator",
                "activity_type",
                "rabbit_type",
                "contributors",
                "project_contributors",
                "dfc_bots",
                "rabbit_bots",
                "agreements",
                "agreement_rate",
            ]
        )

    summary = (
        comparison.groupby(["incubator", "activity_type", "rabbit_type"], dropna=False)
        .agg(
            contributors=("contributor", "nunique"),
            project_contributors=("contributor", "size"),
            dfc_bots=("dfc_is_bot", "sum"),
            rabbit_bots=("rabbit_is_bot", "sum"),
            agreements=("agreement", "sum"),
        )
        .reset_index()
    )
    summary["agreement_rate"] = summary["agreements"] / summary["project_contributors"]

    overall = (
        comparison.groupby("incubator", dropna=False)
        .agg(
            contributors=("contributor", "nunique"),
            project_contributors=("contributor", "size"),
            dfc_bots=("dfc_is_bot", "sum"),
            rabbit_bots=("rabbit_is_bot", "sum"),
            agreements=("agreement", "sum"),
        )
        .reset_index()
    )
    overall["activity_type"] = "all"
    overall["rabbit_type"] = "all"
    overall["agreement_rate"] = overall["agreements"] / overall["project_contributors"]

    return pd.concat([overall[summary.columns], summary], ignore_index=True)


def summary_statistics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compute concise alignment statistics per incubator and overall."""

    columns = [
        "incubator",
        "contributors",
        "project_contributors",
        "agreements",
        "disagreements",
        "alignment_pct",
        "dfc_bots",
        "rabbit_bots",
        "dfc_bot_pct",
        "rabbit_bot_pct",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        comparison.groupby("incubator", dropna=False)
        .agg(
            contributors=("contributor", "nunique"),
            project_contributors=("contributor", "size"),
            agreements=("agreement", "sum"),
            dfc_bots=("dfc_is_bot", "sum"),
            rabbit_bots=("rabbit_is_bot", "sum"),
        )
        .reset_index()
    )

    overall = pd.DataFrame(
        [
            {
                "incubator": "all",
                "contributors": comparison["contributor"].nunique(),
                "project_contributors": comparison.shape[0],
                "agreements": int(comparison["agreement"].sum()),
                "dfc_bots": int(comparison["dfc_is_bot"].sum()),
                "rabbit_bots": int(comparison["rabbit_is_bot"].sum()),
            }
        ]
    )

    stats = pd.concat([grouped, overall], ignore_index=True)
    stats["disagreements"] = stats["project_contributors"] - stats["agreements"]
    stats["alignment_pct"] = stats["agreements"] / stats["project_contributors"] * 100
    stats["dfc_bot_pct"] = stats["dfc_bots"] / stats["project_contributors"] * 100
    stats["rabbit_bot_pct"] = (
        stats["rabbit_bots"] / stats["project_contributors"] * 100
    )

    numeric_cols = ["alignment_pct", "dfc_bot_pct", "rabbit_bot_pct"]
    stats[numeric_cols] = stats[numeric_cols].round(2)
    return stats[columns]


def print_summary_statistics(comparison: pd.DataFrame) -> None:
    """Print concise alignment statistics."""

    stats = summary_statistics(comparison)
    print("\nSummary Statistics")
    if stats.empty:
        print("No comparison rows were generated.")
        return

    print(stats.to_string(index=False))


def benchmark_incubator(
    incubator: str,
    params: dict[str, Any],
    github_api_key: str,
    rabbit_cache_dir: str | Path = "reports/rabbit",
    refresh_rabbit: bool = False,
    min_events: int = 5,
    min_confidence: float = 1.0,
    max_queries: int = 3,
    no_wait: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load rawdata, run RABBIT, attach labels, and compare one incubator."""

    data_lookup = load_rawdata(incubator=incubator)
    rabbit_cache_path = Path(rabbit_cache_dir) / f"{incubator}.csv"
    cached_rabbit = rabbit_cache_path.exists() and not refresh_rabbit

    if cached_rabbit:
        rabbit_predictions = _read_rabbit_csv(rabbit_cache_path)
    else:
        contributors = rawdata_contributors(
            data_lookup=data_lookup,
            incubator=incubator,
            params=params,
        )
        rabbit_predictions = run_rabbit_predictions(
            contributors=contributors,
            github_api_key=github_api_key,
            incubator=incubator,
            min_events=min_events,
            min_confidence=min_confidence,
            max_queries=max_queries,
            no_wait=no_wait,
        )
        check_path(rabbit_cache_path)
        rabbit_predictions.to_csv(rabbit_cache_path, index=False)

    enriched = attach_rabbit_predictions(
        data_lookup=data_lookup,
        incubator=incubator,
        rabbit_predictions=rabbit_predictions,
        params=params,
    )
    comparison = rawdata_to_comparison_rows(
        data_lookup=enriched,
        incubator=incubator,
        params=params,
    )
    return comparison, summarize_comparison(comparison)


def benchmark_all_incubators(
    incubator: str | None = None,
    rabbit_cache_dir: str | Path = "reports/rabbit",
    refresh_rabbit: bool = False,
    output_dir: str | Path = "reports/bot_id",
    min_events: int = 5,
    min_confidence: float = 1.0,
    max_queries: int = 3,
    no_wait: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the bot benchmark for every incubator in params.json."""

    params = load_params()
    github_api_key = load_dotenv_api_key()
    comparisons: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    statuses: list[dict[str, Any]] = []
    incubators = [incubator] if incubator is not None else params["datasets"]
    unknown = set(incubators) - set(params["datasets"])
    if unknown:
        raise ValueError(f"Unknown incubator(s): {sorted(unknown)}")

    for incubator_name in incubators:
        try:
            comparison, summary = benchmark_incubator(
                incubator=incubator_name,
                params=params,
                github_api_key=github_api_key,
                rabbit_cache_dir=rabbit_cache_dir,
                refresh_rabbit=refresh_rabbit,
                min_events=min_events,
                min_confidence=min_confidence,
                max_queries=max_queries,
                no_wait=no_wait,
            )
            comparisons.append(comparison)
            summaries.append(summary)
            statuses.append(
                {
                    "incubator": incubator_name,
                    "status": "ok",
                    "rabbit_results": str(
                        Path(rabbit_cache_dir) / f"{incubator_name}.csv"
                    ),
                    "rows": comparison.shape[0],
                    "error": "",
                }
            )
        except Exception as exc:
            log(f"Skipping {incubator_name}: {exc}", "warning")
            statuses.append(
                {
                    "incubator": incubator_name,
                    "status": "error",
                    "rabbit_results": str(
                        Path(rabbit_cache_dir) / f"{incubator_name}.csv"
                    ),
                    "rows": 0,
                    "error": str(exc),
                }
            )

    comparison_all = (
        pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()
    )
    summary_all = (
        pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    )
    status_df = pd.DataFrame.from_records(statuses)

    output_dir = Path(output_dir)
    detail_path = output_dir / "bot_id_comparison.csv"
    summary_path = output_dir / "bot_id_summary.csv"
    status_path = output_dir / "bot_id_status.csv"
    check_path(detail_path)
    comparison_all.to_csv(detail_path, index=False)
    summary_all.to_csv(summary_path, index=False)
    status_df.to_csv(status_path, index=False)

    return comparison_all, summary_all, status_df


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare DFC rawdata bot labels against RABBIT predictions "
            "for every incubator in ref/params.json."
        )
    )
    parser.add_argument(
        "--incubator",
        help=(
            "Optional incubator to run. Defaults to every incubator listed in "
            "ref/params.json."
        ),
    )
    parser.add_argument(
        "--rabbit-cache-dir",
        default="reports/rabbit",
        help="Directory where RABBIT CSV outputs are cached.",
    )
    parser.add_argument(
        "--refresh-rabbit",
        action="store_true",
        help="Re-run RABBIT even when cached incubator CSVs exist.",
    )
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=1.0)
    parser.add_argument("--max-queries", type=int, default=3)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--output-dir", default="reports/bot_id_benchmark")
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    comparison, summary, status = benchmark_all_incubators(
        incubator=args.incubator,
        rabbit_cache_dir=args.rabbit_cache_dir,
        refresh_rabbit=args.refresh_rabbit,
        output_dir=args.output_dir,
        min_events=args.min_events,
        min_confidence=args.min_confidence,
        max_queries=args.max_queries,
        no_wait=args.no_wait,
    )

    print(status.to_string(index=False))
    if not summary.empty:
        print()
        print(summary.to_string(index=False))

    print_summary_statistics(comparison)


if __name__ == "__main__":
    main()
