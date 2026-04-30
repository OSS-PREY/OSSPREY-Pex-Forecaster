"""
Benchmark DFC rawdata aliases against Gambit disambiguation.

1. Load RawData with its default versions; the data must already contain DFC's
   dealiased author field.
2. Build Gambit alias inputs from rawdata author names and emails.
3. Run ``gambit.main.disambiguate_aliases``.
4. Join Gambit's disambiguated names back onto rawdata rows.
5. Compare DFC dealiased names against Gambit names.
6. Sample one project per incubator and repeat for every incubator listed in
   ``ref/params.json`` unless one incubator is requested.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from gambit import main as gambit
from dfc.abstractions.rawdata import RawData
from dfc.utils import check_path, load_params, log


DFC_ALIAS_FIELDS = (
    "dealised_author_full_name",
    "dealiased_author_full_name",
)
EMAIL_FIELDS = (
    "sender_email",
    "author_email",
    "email"
)
GAMBIT_NAME_FIELDS = (
    "disambiguated_name",
    "gambit_name",
    "canonical_name",
    "canonical_alias",
    "identity_name",
    "name",
)
GAMBIT_ID_FIELDS = (
    "disambiguated_id",
    "gambit_id",
    "canonical_id",
    "identity_id",
    "person_id",
    "cluster_id",
)
DEFAULT_SAMPLE_SEED = 0
BOT_FIELD = "is_bot"


def _first_existing(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate present in columns, case-insensitively."""

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


def sample_one_project(
    data_lookup: dict[str, pd.DataFrame],
    incubator: str,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, pd.DataFrame]:
    """Return rawdata filtered to one seeded random non-bot project subset."""

    project_frames = [
        df["project_name"].dropna().astype(str)
        for df in data_lookup.values()
        if "project_name" in df.columns
    ]
    project_names = (
        pd.concat(project_frames, ignore_index=True).drop_duplicates()
        if project_frames
        else pd.Series(dtype=str)
    )
    if project_names.empty:
        raise ValueError(f"No project_name values found for {incubator}")

    sampled_project = project_names.sample(n=1, random_state=sample_seed).iloc[0]
    log(f"{incubator}: sampled project {sampled_project}", "note")
    filtered: dict[str, pd.DataFrame] = {}
    for activity_type, df in data_lookup.items():
        if "project_name" not in df.columns:
            filtered[activity_type] = df.iloc[0:0].copy()
            continue

        project_mask = df["project_name"].astype(str) == sampled_project
        bot_mask = (
            df[BOT_FIELD] == 1
            if BOT_FIELD in df.columns
            else pd.Series(False, index=df.index)
        )
        filtered[activity_type] = df[project_mask & ~bot_mask].copy()

    return filtered


def _alias_fields(df: pd.DataFrame, author_field: str) -> tuple[str, str, str]:
    """Resolve raw author, email, and DFC dealiased fields."""

    columns = set(df.columns)
    name_field = author_field if author_field in columns else None
    email_field = _first_existing(columns, EMAIL_FIELDS)
    dfc_alias_field = _first_existing(columns, DFC_ALIAS_FIELDS)

    if name_field is None:
        raise ValueError(f"missing author field {author_field!r}")
    if dfc_alias_field is None:
        raise ValueError(f"missing DFC dealiased field; tried {DFC_ALIAS_FIELDS}")

    return name_field, email_field or name_field, dfc_alias_field


def build_alias_inputs(
    data_lookup: dict[str, pd.DataFrame],
    incubator: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Build unique alias_name/alias_email pairs for Gambit."""

    author_field = params["author-source-field"][incubator]
    alias_frames: list[pd.DataFrame] = []

    for activity_type, df in data_lookup.items():
        if df.empty:
            continue

        name_field, email_field, _ = _alias_fields(df, author_field)
        aliases = pd.DataFrame(
            {
                "alias_name": df[name_field].astype(str),
                "alias_email": df[email_field].astype(str),
            }
        )
        aliases = aliases[
            aliases["alias_name"].str.strip().ne("")
            & aliases["alias_name"].str.lower().ne("none")
        ]
        aliases["activity_type"] = activity_type
        alias_frames.append(aliases)

    if not alias_frames:
        return pd.DataFrame(columns=["alias_name", "alias_email"])

    aliases = pd.concat(alias_frames, ignore_index=True)
    return aliases[["alias_name", "alias_email"]].drop_duplicates().reset_index(drop=True)


def normalize_gambit_output(aliases: pd.DataFrame, gambit_output: Any) -> pd.DataFrame:
    """Normalize Gambit's output to alias_name/alias_email/gambit_* columns."""

    if isinstance(gambit_output, pd.DataFrame):
        result = gambit_output.copy()
    elif isinstance(gambit_output, pd.Series):
        result = aliases.copy()
        result["gambit_disambiguated_name"] = gambit_output.reset_index(drop=True)
    else:
        result = pd.DataFrame(gambit_output)

    if result.shape[0] != aliases.shape[0]:
        raise ValueError(
            "Gambit output row count does not match alias input row count: "
            f"{result.shape[0]} != {aliases.shape[0]}"
        )

    result = result.reset_index(drop=True)
    aliases = aliases.reset_index(drop=True)
    for col in ("alias_name", "alias_email"):
        if col not in result.columns:
            result[col] = aliases[col]

    name_col = _first_existing(set(result.columns), GAMBIT_NAME_FIELDS)
    id_col = _first_existing(set(result.columns), GAMBIT_ID_FIELDS)

    if name_col is None:
        extra_cols = [
            col for col in result.columns
            if col not in {"alias_name", "alias_email"}
        ]
        if extra_cols:
            name_col = extra_cols[0]
        else:
            name_col = "alias_name"

    normalized = result[["alias_name", "alias_email"]].copy()
    normalized["gambit_disambiguated_name"] = result[name_col].astype(str)
    if id_col is not None:
        normalized["gambit_disambiguated_id"] = result[id_col]

    return normalized.drop_duplicates(subset=["alias_name", "alias_email"], keep="last")


def run_gambit_aliases(aliases: pd.DataFrame, incubator: str) -> pd.DataFrame:
    """Run Gambit disambiguation on alias_name/alias_email pairs."""

    if aliases.empty:
        return pd.DataFrame(
            columns=["alias_name", "alias_email", "gambit_disambiguated_name"]
        )

    tqdm.pandas(desc=f"{incubator}: Gambit aliases")
    aliases = aliases.copy()
    aliases["alias_name"] = aliases["alias_name"].progress_apply(str)
    result = gambit.disambiguate_aliases(aliases[["alias_name", "alias_email"]])
    return normalize_gambit_output(aliases, result)


def _read_gambit_cache(path: str | Path) -> pd.DataFrame:
    """Read cached Gambit predictions."""

    return pd.read_csv(path)


def attach_gambit_aliases(
    data_lookup: dict[str, pd.DataFrame],
    incubator: str,
    gambit_aliases: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Join Gambit disambiguation onto rawdata rows."""

    author_field = params["author-source-field"][incubator]
    enriched: dict[str, pd.DataFrame] = {}

    for activity_type, df in data_lookup.items():
        if df.empty:
            enriched[activity_type] = df.copy()
            continue

        name_field, email_field, dfc_alias_field = _alias_fields(df, author_field)
        out = df.copy()
        tqdm.pandas(desc=f"{incubator}: attach {activity_type}")
        out["_gambit_alias_name"] = out[name_field].progress_apply(str)
        out["_gambit_alias_email"] = out[email_field].astype(str)
        out["_dfc_dealiased_name"] = out[dfc_alias_field].astype(str)

        out = out.merge(
            gambit_aliases,
            left_on=["_gambit_alias_name", "_gambit_alias_email"],
            right_on=["alias_name", "alias_email"],
            how="left",
        )
        out["gambit_disambiguated_name"] = (
            out["gambit_disambiguated_name"].fillna(out["_gambit_alias_name"])
        )
        out.drop(columns=["alias_name", "alias_email"], inplace=True)
        enriched[activity_type] = out

    return enriched


def rawdata_to_comparison_rows(
    data_lookup: dict[str, pd.DataFrame],
    incubator: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Aggregate enriched rawdata into alias comparison rows."""

    author_field = params["author-source-field"][incubator]
    records: list[dict[str, Any]] = []

    for activity_type, df in data_lookup.items():
        if df.empty:
            continue

        required = {
            "project_name",
            author_field,
            "_gambit_alias_name",
            "_gambit_alias_email",
            "_dfc_dealiased_name",
            "gambit_disambiguated_name",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{activity_type} enriched rawdata for {incubator} is missing "
                f"columns: {sorted(missing)}"
            )

        grouped = (
            df.groupby(
                [
                    "project_name",
                    "_gambit_alias_name",
                    "_gambit_alias_email",
                    "_dfc_dealiased_name",
                    "gambit_disambiguated_name",
                ],
                observed=True,
            )
            .size()
            .rename("activity_count")
            .reset_index()
        )

        def build_record(row: pd.Series) -> dict[str, Any]:
            return {
                "incubator": incubator,
                "project_name": row["project_name"],
                "activity_type": activity_type,
                "alias_name": row["_gambit_alias_name"],
                "alias_email": row["_gambit_alias_email"],
                "dfc_dealiased_name": row["_dfc_dealiased_name"],
                "gambit_disambiguated_name": row["gambit_disambiguated_name"],
                "activity_count": int(row["activity_count"]),
            }

        tqdm.pandas(desc=f"{incubator}: compare {activity_type}")
        records.extend(grouped.progress_apply(build_record, axis=1).tolist())

    comparison = pd.DataFrame.from_records(records)
    if comparison.empty:
        comparison["agreement"] = pd.Series(dtype="int64")
    else:
        dfc_names = comparison["dfc_dealiased_name"].str.replace(
            r"\s*\([^)]*\)",
            "",
            regex=True,
        )
        gambit_names = comparison["gambit_disambiguated_name"].str.replace(
            r"\s*\([^)]*\)",
            "",
            regex=True,
        )
        comparison["agreement"] = (
            dfc_names.str.casefold() == gambit_names.str.casefold()
        ).astype(int)
    return comparison


def summarize_comparison(
    comparison: pd.DataFrame,
    incubator: str | None = None,
    activity_types: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize DFC-vs-Gambit alias agreement."""

    columns = [
        "incubator",
        "activity_type",
        "aliases",
        "rows",
        "agreements",
        "agreement_rate",
        "dfc_names",
        "gambit_names",
    ]

    def zero_row(incubator_name: str, activity_type: str) -> dict[str, Any]:
        return {
            "incubator": incubator_name,
            "activity_type": activity_type,
            "aliases": 0,
            "rows": 0,
            "agreements": 0,
            "agreement_rate": 1.0,
            "dfc_names": 0,
            "gambit_names": 0,
        }

    if comparison.empty:
        if incubator is None:
            return pd.DataFrame(columns=columns)

        rows = [zero_row(incubator, "all")]
        rows.extend(
            zero_row(incubator, activity_type)
            for activity_type in (activity_types or [])
        )
        return pd.DataFrame.from_records(rows, columns=columns)

    summary = (
        comparison.groupby(["incubator", "activity_type"], dropna=False)
        .agg(
            aliases=("alias_name", "nunique"),
            rows=("alias_name", "size"),
            agreements=("agreement", "sum"),
            dfc_names=("dfc_dealiased_name", "nunique"),
            gambit_names=("gambit_disambiguated_name", "nunique"),
        )
        .reset_index()
    )
    summary["agreement_rate"] = summary["agreements"] / summary["rows"]

    overall = (
        comparison.groupby("incubator", dropna=False)
        .agg(
            aliases=("alias_name", "nunique"),
            rows=("alias_name", "size"),
            agreements=("agreement", "sum"),
            dfc_names=("dfc_dealiased_name", "nunique"),
            gambit_names=("gambit_disambiguated_name", "nunique"),
        )
        .reset_index()
    )
    overall["activity_type"] = "all"
    overall["agreement_rate"] = overall["agreements"] / overall["rows"]

    if activity_types:
        existing_activity_types = set(summary["activity_type"])
        missing_rows = [
            zero_row(incubator_name, activity_type)
            for incubator_name in comparison["incubator"].dropna().unique()
            for activity_type in activity_types
            if activity_type not in existing_activity_types
        ]
        if missing_rows:
            summary = pd.concat(
                [summary, pd.DataFrame.from_records(missing_rows)],
                ignore_index=True,
            )

    return pd.concat([overall[summary.columns], summary], ignore_index=True)


def summary_statistics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compute concise alias alignment statistics per incubator and overall."""

    columns = [
        "incubator",
        "aliases",
        "rows",
        "agreements",
        "disagreements",
        "alignment_pct",
        "dfc_names",
        "gambit_names",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        comparison.groupby("incubator", dropna=False)
        .agg(
            aliases=("alias_name", "nunique"),
            rows=("alias_name", "size"),
            agreements=("agreement", "sum"),
            dfc_names=("dfc_dealiased_name", "nunique"),
            gambit_names=("gambit_disambiguated_name", "nunique"),
        )
        .reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "incubator": "all",
                "aliases": comparison["alias_name"].nunique(),
                "rows": comparison.shape[0],
                "agreements": int(comparison["agreement"].sum()),
                "dfc_names": comparison["dfc_dealiased_name"].nunique(),
                "gambit_names": comparison["gambit_disambiguated_name"].nunique(),
            }
        ]
    )

    stats = pd.concat([grouped, overall], ignore_index=True)
    stats["disagreements"] = stats["rows"] - stats["agreements"]
    stats["alignment_pct"] = (stats["agreements"] / stats["rows"] * 100).round(2)
    return stats[columns]


def print_summary_statistics(comparison: pd.DataFrame) -> None:
    """Print concise alias alignment statistics."""

    stats = summary_statistics(comparison)
    print("\nAlias Summary Statistics")
    if stats.empty:
        print("No comparison rows were generated.")
        return

    print(stats.to_string(index=False))


def final_alignment_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the final alias alignment report from the summary table."""

    columns = [
        "section",
        "incubator",
        "activity_type",
        "alignment_pct",
        "median_alignment_pct",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)

    activity_summary = summary[summary["activity_type"] != "all"].copy()
    if activity_summary.empty:
        return pd.DataFrame(columns=columns)

    def weighted_median(values: pd.Series, weights: pd.Series) -> float:
        valid = values.notna() & weights.notna() & weights.gt(0)
        if not valid.any():
            return float(values.median()) if not values.empty else 0.0

        sorted_values = values[valid].sort_values()
        sorted_weights = weights[valid].loc[sorted_values.index]
        cutoff = sorted_weights.sum() / 2
        return float(sorted_values[sorted_weights.cumsum() >= cutoff].iloc[0])

    incubator_breakdown = activity_summary[
        [
            "incubator",
            "activity_type",
            "rows",
            "agreements",
            "agreement_rate",
        ]
    ].copy()
    incubator_breakdown["alignment_pct"] = (
        incubator_breakdown["agreement_rate"].mul(100).round(2)
    )
    incubator_breakdown["section"] = "incubator"

    overall = (
        activity_summary.groupby("activity_type", dropna=False)
        .agg(alignment_pct=("agreement_rate", lambda values: values.mean() * 100))
        .reset_index()
    )
    overall["alignment_pct"] = overall["alignment_pct"].round(2)
    overall["section"] = "overall"
    overall["incubator"] = "all"
    median_alignment = weighted_median(
        incubator_breakdown["alignment_pct"],
        incubator_breakdown["rows"],
    )

    report = pd.concat(
        [
            incubator_breakdown[
                ["section", "incubator", "activity_type", "alignment_pct"]
            ],
            overall[["section", "incubator", "activity_type", "alignment_pct"]],
        ],
        ignore_index=True,
    )
    report["median_alignment_pct"] = pd.NA
    overall_mask = report["section"] == "overall"
    report.loc[overall_mask, "median_alignment_pct"] = round(median_alignment, 2)
    return report[columns]


def benchmark_incubator(
    incubator: str,
    params: dict[str, Any],
    gambit_cache_dir: str | Path = "reports/gambit",
    refresh_gambit: bool = False,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load rawdata, run Gambit, attach names, and compare one incubator."""

    data_lookup = sample_one_project(
        data_lookup=load_rawdata(incubator=incubator),
        incubator=incubator,
        sample_seed=sample_seed,
    )
    gambit_cache_path = Path(gambit_cache_dir) / f"{incubator}.csv"

    if gambit_cache_path.exists() and not refresh_gambit:
        gambit_aliases = _read_gambit_cache(gambit_cache_path)
    else:
        aliases = build_alias_inputs(
            data_lookup=data_lookup,
            incubator=incubator,
            params=params,
        )
        gambit_aliases = run_gambit_aliases(aliases, incubator=incubator)
        check_path(gambit_cache_path)
        gambit_aliases.to_csv(gambit_cache_path, index=False)

    enriched = attach_gambit_aliases(
        data_lookup=data_lookup,
        incubator=incubator,
        gambit_aliases=gambit_aliases,
        params=params,
    )
    comparison = rawdata_to_comparison_rows(
        data_lookup=enriched,
        incubator=incubator,
        params=params,
    )
    return comparison, summarize_comparison(
        comparison,
        incubator=incubator,
        activity_types=list(data_lookup.keys()),
    )


def benchmark_all_incubators(
    incubator: str | None = None,
    gambit_cache_dir: str | Path = "reports/gambit",
    refresh_gambit: bool = False,
    output_dir: str | Path = "reports/alias_benchmark",
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the alias benchmark for one sampled project per incubator."""

    params = load_params()
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
                gambit_cache_dir=gambit_cache_dir,
                refresh_gambit=refresh_gambit,
                sample_seed=sample_seed,
            )
            comparisons.append(comparison)
            summaries.append(summary)
            statuses.append(
                {
                    "incubator": incubator_name,
                    "status": "ok",
                    "gambit_results": str(
                        Path(gambit_cache_dir) / f"{incubator_name}.csv"
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
                    "gambit_results": str(
                        Path(gambit_cache_dir) / f"{incubator_name}.csv"
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
    detail_path = output_dir / "alias_comparison.csv"
    summary_path = output_dir / "alias_summary.csv"
    status_path = output_dir / "alias_status.csv"
    final_report_path = output_dir / "alias_final_report.csv"
    check_path(detail_path)
    comparison_all.to_csv(detail_path, index=False)
    summary_all.to_csv(summary_path, index=False)
    status_df.to_csv(status_path, index=False)
    final_alignment_report(summary_all).to_csv(final_report_path, index=False)

    return comparison_all, summary_all, status_df


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare DFC rawdata dealiased names against Gambit aliases "
            "for one sampled project per incubator in ref/params.json."
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
        "--gambit-cache-dir",
        default="reports/gambit",
        help="Directory where Gambit CSV outputs are cached.",
    )
    parser.add_argument(
        "--refresh-gambit",
        action="store_true",
        help="Re-run Gambit even when cached incubator CSVs exist.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Random seed used to sample one project per incubator.",
    )
    parser.add_argument("--output-dir", default="reports/alias_benchmark")
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    comparison, summary, status = benchmark_all_incubators(
        incubator=args.incubator,
        gambit_cache_dir=args.gambit_cache_dir,
        refresh_gambit=args.refresh_gambit,
        output_dir=args.output_dir,
        sample_seed=args.sample_seed,
    )

    print(status.to_string(index=False))
    if not summary.empty:
        print()
        print(summary.to_string(index=False))

    print_summary_statistics(comparison)


if __name__ == "__main__":
    main()
