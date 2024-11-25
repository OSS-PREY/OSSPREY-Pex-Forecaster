# Project Explorer (PEx) Forecasting
This codebase is a modified (streamlined, simplified) pipeline based on the
deeper research done for sustainability forecasting.

## Installation
### Cloning the Repository
To get started, first clone the pipeline via:
```
git clone https://github.com/arjashok/pex-forecaster.git
```

The directory structure is as follows:
```
pex-forecaster/
│   README.md
│   requirements.txt    
│
└───src/
|   |   utils.py
|   |
│   └───abstractions/
│   │       |   ## contains all the abstractions needed for the pipeline
│   │       |   modeldata.py
│   │       |   netdata.py
│   │       |   perfdata.py
│   │       |   rawdata.py
│   │       |   tsmodel.py
│   │
|   └───algorithms/
│   │       |   ## algorithms for forecasting and other pipeline adjacent tasks
│   │       |   trajectory.py
│   │
│   └───pipeline/
│       │   ## core pipeline functionality from rawdata --> net & model outputs
│       │   compile_data.py
│       │   create_networks.py
│       │   modeling.py
│       │   monthly_data.py
│       │   network_features.py
│       │   network_visualizations.py
│       │   pipeline.py
│   
└───ref/
|   │   ## various reference materials for the programs to use; DO NOT MODIFY
|   │   ## ANY FILES except for 'params.json'
|
└───data/
    │   ## data should be contained here, see following section for more info...
```

### Dataset Retrieval
To setup the datasets locally, ensure the following directory structure within
`/pex-forecaster/data/` is met:
```
data/
│   README.md
│   requirements.txt    
│
└───apache_data/
│   |   commits.csv (01-commits-clean.csv)
│   |   emails.csv (01-emails-clean.csv)
|
└───github_data/
│   |   commits.csv (03-commits-dealias.csv)
│   |   issues.csv (04-issues-dealias.csv)
│
└───eclipse_data/
    |   commits.csv (01-commits-processed.csv)
    |   issues.csv (01-issues-processed.csv)
```

There are various ways to get started with further unprocessed data or without 
raw data at all, but given the narrow focus of this pipeline we will assume 
only the most processed data is ready. These are available at the following
links:

NOT YET UPLOADED, WORK IN PROGRESS...
- [apache_data]()
- [eclipse_data]()
- [github_data]()


### Environment Setup
It's recommended to use virtual environments to run this module, although most
environments with relatively modern versions of the listed modules in
`requirements.txt` likely will suffice. We've elected to use Conda. Install all 
requirements (or check the installation) via:
```
pip install -r requirements.txt
```

## Usage
All modules under `src` are written such that they can be individually imported
into another program without too much dependency on the narrow task, i.e.
somewhat package-ready. That said, to perform the tasks this module is built
for, do the following:

1. Raw Data --> Network Data Pipeline: leverage the `src/pipeline/` module to 
    run the tasks required; by default, the `pipeline.py` wrapper calls all 
    required functions to do this, including some helpful utility to pre-process
    raw data as well--given we assume pre-processed data is already being used,
    we'll ignore that functionality.

2. Network Data --> Sustainability Forecasts: leverage the `src/pipeline/`
    module once again to do this; specifically, `modeling.py` is focused on
    building different types of trials with helpful functions for individual
    tests. For a more extensive overview of the grammar we use to parse a
    trial's goal, reference the documentation found in the sister-research 
    repository [here](https://github.com/arjashok/OSS-Research)

3. Trajectory Generation: the module `src/algorithms/trajectory.py` focuses 
    entirely on generating the future trajectories. Further documentation on the
    strategies and functionality are present within the program itself.

Quick usage commands are provided here:
- Modeling:
```
python3 "_modeling.py" '{"strategy": "Acbn + Ecbn^ --> Ecbn^^", "trials": 5, "model-arch": "BLSTM", "hyperparams": {"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}}'
```

- Network Gen:
```
python3 "pipeline.py" '{"incubator": "apache", "versions": {"tech": 1, "social": 1}}'
```

- Visualizations:
```
python3 "perfdata.py"
```


## Mechanisms
Some useful, but constrictive, mechanisms are baked into the pipeline such as:

1. Caching: network data is all cached into a directory named `network_data`;
    all augmentations are similarly cahced along with it. While this is in the
    process of being changed to work on the fly, we encourage users to preserve
    only the clean copies of the network data at any time should there be any 
    issues with coherency.

2. Directory: admittedly, the directory pathing is in disarray right now; while
    this is being fixed overtime, we encourage abiding by the directory
    structure given to avoid downstream errors and potential hidden 
    miscalculations. **We also encourage runnning programs explicitly from the 
    `/pex-forecaster/` directory itself since all relative imports are being 
    structured to mimic that. In future versions, we'll attempt to remedy this 
    to enable easier package use and program running. Please leverage the 
    example usage program in the meantime.


## Repository Maintainance
Please feel free to report errors/bugs, suggest features, or reach out with any
questions/concerns. The primary maintainer is Arjun (arjun3.ashok@gmail.com,
arjashok@ucdavis.edu).

