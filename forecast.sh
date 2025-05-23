## Forecasting Example Scripts (training + evaluation)
python3 -m decalforecaster.pipeline.modeling --kwargs \
    strategy="A^ --> A^^" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

python3 -m decalforecaster.pipeline.modeling --kwargs \
    strategy="E^ --> E^^" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

python3 -m decalforecaster.pipeline.modeling --kwargs \
    strategy="A + E --> G" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

