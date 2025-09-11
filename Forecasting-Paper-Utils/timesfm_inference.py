from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

try:
    import timesfm
except Exception as exc:  # pragma: no cover - timesfm may be unavailable
    timesfm = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _load_model() -> "timesfm.TimesFm":
    """Load a pre-trained TimesFM model for inference.

    Returns
    -------
    timesfm.TimesFm
        Loaded TimesFM model. If the library is unavailable, an ImportError is
        raised.
    """
    if timesfm is None:
        raise ImportError(
            "timesfm library is not installed or failed to import"
        ) from _IMPORT_ERROR

    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu", per_core_batch_size=1, horizon_len=1
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )


def _forecast_series(model: "timesfm.TimesFm", series: Iterable[float]) -> float:
    """Forecast the next value of a time series using TimesFM.

    Parameters
    ----------
    model:
        Loaded TimesFM model.
    series:
        Iterable of numeric values representing the time series history.

    Returns
    -------
    float
        Forecasted next value of the series.
    """
    array = np.asarray(list(series), dtype=float)
    # TimesFM expects a list of arrays for batch forecasting.
    point_forecast, _ = model.forecast([array])
    # `point_forecast` is (batch, horizon_len)
    return float(point_forecast[0][0])


def run_inference(feature_path: Path, target_path: Path, output_path: Path,
                  feature_col: str = "s_num_nodes") -> pd.DataFrame:
    """Run TimesFM inference for a single foundation dataset.

    Parameters
    ----------
    feature_path:
        CSV file containing feature data with ``proj_name`` and ``month``.
    target_path:
        CSV file containing project-level targets with columns ``project`` and
        ``target``.
    output_path:
        Path where the prediction CSV will be written.
    feature_col:
        Which feature column to use as the univariate time series for
        forecasting.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing project name, predicted value and target label.
    """
    features = pd.read_csv(feature_path)
    targets = pd.read_csv(target_path)[["project", "target"]]

    try:
        model = _load_model()
    except ImportError:
        model = None

    rows: List[dict] = []
    for project, group in features.groupby("proj_name"):
        history = group.sort_values("month")[feature_col].to_list()
        pred_value = np.nan
        if model is not None and len(history) > 0:
            pred_value = _forecast_series(model, history)
        target_row = targets[targets["project"] == project]
        target_val = target_row["target"].iloc[0] if not target_row.empty else None
        rows.append({"project": project, "prediction": pred_value, "target": target_val})

    result = pd.DataFrame(rows)
    result['predicted_label'] = np.where(result['prediction'] > 0.5, 'graduated', 'retired')
    result.to_csv(output_path, index=False)
    return result


def evaluate_transfer(source: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Evaluate ``target`` predictions with a fixed 0.5 threshold.

    The ``target`` column is treated as binary with ``graduated`` as the
    positive class and all other labels considered negative. Predictions above
    0.5 are classified as ``graduated``.

    Parameters
    ----------
    source:
        Unused DataFrame retained for backward compatibility.
    target:
        DataFrame providing predictions and targets to evaluate.

    Returns
    -------
    dict
        Dictionary with ``precision``, ``recall`` and ``f1`` scores.
    """

    y_true = target["target"].map(lambda x: 1 if x == "graduated" else 0).to_numpy()
    tgt_preds = np.nan_to_num(target["prediction"].to_numpy(), nan=0.0)
    y_pred = (tgt_preds > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    datasets = {
        "apache": ("clean-apache-network-data-2-2.csv", "apache_inferences.csv"),
        "eclipse": ("clean-eclipse-network-data-3-3.csv", "eclipse_inferences.csv"),
        "github": ("clean-github-network-data-4-5.csv", "github_inferences.csv"),
        "osgeo": ("clean-osgeo-network-data-2-2.csv", "osgeo_inferences.csv"),
    }

    results = {}
    for name, (feat_file, target_file) in datasets.items():
        results[name] = run_inference(
            base / feat_file,
            base / target_file,
            base / f"{name}_timesfm_predictions.csv",
        )

    for src_name, src_df in results.items():
        for tgt_name, tgt_df in results.items():
            metrics = evaluate_transfer(src_df, tgt_df)
            print(
                f"{src_name}->{tgt_name}: precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
            )
