# Project Explorer (PEx) Forecasting
This codebase is a modified (streamlined, simplified) pipeline based on the
deeper research done for sustainability forecasting.

## Regeneration
To regenerate the complete pipeline and results, run `full_pipeline.sh`:
```bash
bash full_pipeline.sh
```

## Installation
The directory structure is as follows:
```
pex-forecaster/
│   README.md
│   requirements.txt
│   pre_process.sh
│   full_pipeline.sh
│
└───dfc/
|   |   utils.py
|   |   verify.py
|   |
│   └───abstractions/
│   │       │   deltadata.py
│   │       │   modeldata.py
│   │       │   netdata.py
│   │       │   perfdata.py
│   │       │   projdata.py
│   │       │   rawdata.py
│   │       │   tsmodel.py
│   │
|   └───algorithms/
│   │       │   trajectory.py
│   │
│   └───pipeline/
│   │       │   compile_data.py
│   │       │   create_networks.py
│   │       │   inference.py
│   │       │   modeling.py
│   │       │   monthly_data.py
│   │       │   network_features.py
│   │       │   network_visualizations.py
│   │       │   pipeline.py
│   │
│   └───scripts/
│       │   combine_social.py
│       │   enforce_column_names.py
│       │   enforce_dates.py
│       │   pre_process.py
│       │   standardize_sender_aliases.py
│       │   triager.py
│
└───ref/
|   │   various reference materials for data processing
|
└───data/
    │   raw and processed datasets
|
└───reports/
    │   processing reports and logs
```

### Dataset Retrieval
To setup the datasets locally, ensure the following directory structure within
`/pex-forecaster/data/` is met:
```
data/
└───apache_data/
│   |   commits.parquet
│   |   emails.parquet
|
└───github_data/
│   |   commits.parquet
│   |   issues.parquet
│
└───eclipse_data/
    |   commits.parquet
    |   issues.parquet
|
└───osgeo_data/
    |   commits.parquet
    |   emails.parquet
    |   issues.parquet
```

These processed datasets are available at the below links. Alternatively, we
recommend running the following script to verify the structure for you:
```
python3 -m dfc.verify
```

- [apache_data](https://drive.google.com/drive/folders/1-f8AEReRwegpecnOXmdg5XdrzZPuULeF?usp=drive_link)
- [eclipse_data](https://drive.google.com/drive/folders/1CNLy-d353_KL0L-QxiUMTOZTpfCj1YSA?usp=drive_link)
- [github_data](https://drive.google.com/drive/folders/1NPa5oBV_e9mduITmXyw_VrxnrmmBXc1e?usp=drive_link)

### Environment Setup
It's recommended to use virtual environments to run this module. Install all
requirements via:
```
pip install -r requirements.txt
```

## Scripts
1. **Pre-process raw data** (optional): Run `pre_process.sh` to clean, standardize, and prepare data for Eclipse and OSGeo incubators:
```bash
bash pre_process.sh
```

1. **Execute full pipeline**: Run `full_pipeline.sh` to generate networks, train models, and produce forecasts:
```bash
bash full_pipeline.sh
```

Individual components can also be invoked directly via Python modules:

- **Network Generation**:
```
python3 -m dfc.pipeline.pipeline --kwargs \
    incubator=apache \
    versions='{"tech": 1, "social": 1}'
```

- **Modeling**:
```
python3 -m dfc.pipeline.modeling --kwargs \
    trial-type="jss" \
    trials=3 \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 100}'
```

- **OSS ProF & Inference**:
```
python3 -m dfc.scripts.triager
```

## Mechanisms
Key considerations for running the pipeline:

1. **Caching**: Network data is cached in the `network_data/` directory. Preserve clean copies to avoid coherency issues.

2. **Directory Structure**: All relative imports assume execution from the `/pex-forecaster/` root directory. Run all commands from this location.

3. **Centralized Parameters**: Core parameters are stored in `/pex-forecaster/ref/params.json` and shared across pipeline stages.

