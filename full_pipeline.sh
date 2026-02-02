# Full Pipeline Script #


## STEP I: Retrieve Raw Data & Pre-Process ##
python3 -m dfc.verify # retrieve
bash pre_process.sh # adjustments


# STEP II: Socio-Technical Network Generation
python3 -m dfc.pipeline.pipeline --kwargs \
    incubator=apache \
    versions='{"tech": 1, "social": 1}'

python3 -m dfc.pipeline.pipeline --kwargs \
    incubator=github \
    versions='{"tech": 3, "social": 4}'

python3 -m dfc.pipeline.pipeline --kwargs \
    incubator=eclipse \
    versions='{"tech": 2, "social": 2}'

python3 -m dfc.pipeline.pipeline --kwargs \
    incubator=osgeo \
    versions='{"tech": 2, "social": 2}'


## STEP III: Modeling Trials
# generate trials
python3 -m dfc.pipeline.modeling --kwargs \
    trial-type="jss" \
    trials=3 \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 100, "hidden_size": 64, "num_layers": 2, "dropout_rate": 0.5}'

# summarize
python3 -m dfc.abstractions.perfdata --kwargs \
    breakdown-type="jss" \
    acc_measure="mic-f1"


## STEP IV: OSS-ProF
python3 -m dfc.scripts.triager


## STEP V: Manual Compare of Triage Table vs Model
## using the triage output (also see triage_inference/triage_results.ipynb), we
## can compute the triage's accuracy using the best models from the modeling
## stage

