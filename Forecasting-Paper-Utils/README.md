# Forecasting Paper Utilities

This directory collects datasets and scripts used to evaluate sustainability forecasting models.

## TimesFM Inference

The `timesfm_inference.py` script runs zero-shot [TimesFM](https://github.com/google-research/timesfm) predictions on project-level time-series features.

### Requirements

* Python 3.10+
* [timesfm](https://pypi.org/project/timesfm/) (optional). If the library is not installed, predictions are recorded as `NaN` and evaluation metrics will be zero.

Install dependencies with:

```bash
pip install -r ../requirements.txt
pip install timesfm  # optional
```

### Usage

Execute the script from the repository root to generate predictions and evaluate cross-dataset transfer:

```bash
python Forecasting-Paper-Utils/timesfm_inference.py
```

The script forecasts the next value of the `s_num_nodes` feature for each project in the Apache, Eclipse, GitHub, and OSGeo datasets and writes prediction CSVs named `<foundation>_timesfm_predictions.csv`. It also prints precision, recall and F1-scores for every source→target dataset combination using a 0.5 threshold.

### Data Files

The script expects the following files in this directory:

| Dataset | Feature CSV | Target CSV |
|---------|-------------|------------|
| Apache  | `clean-apache-network-data-2-2.csv` | `apache_inferences.csv` |
| Eclipse | `clean-eclipse-network-data-3-3.csv` | `eclipse_inferences.csv` |
| GitHub  | `clean-github-network-data-4-5.csv` | `github_inferences.csv` |
| OSGeo   | `clean-osgeo-network-data-2-2.csv` | `osgeo_inferences.csv` |

Each target file must contain `project` and `target` columns. The feature files must include `proj_name`, `month` and `s_num_nodes` columns.

