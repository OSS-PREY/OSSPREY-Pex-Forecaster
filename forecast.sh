# Forecasting Example Scripts (training + evaluation)
# python3 -m decalfc.pipeline.modeling --kwargs \
#     strategy="A-1-1^ --> A-1-1^^" \
#     trials=1 \
#     model-arch="BLSTM" \
#     hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

# python3 -m decalfc.pipeline.modeling --kwargs \
#     strategy="A-1-1^ --> A-1-1^^" \
#     trials=1 \
#     model-arch="Transformer" \
#     hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

# python3 -m decalfc.pipeline.modeling --kwargs \
#     strategy="E-3-3^ --> E-3-3^^" \
#     trials=1 \
#     model-arch="Transformer" \
#     hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

# python3 -m decalfc.pipeline.modeling --kwargs \
#     strategy="A --> E" \
#     trials=1 \
#     model-arch="Transformer" \
#     hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

# python3 -m decalfc.pipeline.modeling --kwargs \
#     strategy="A + E --> G" \
#     trials=1 \
#     model-arch="BLSTM" \
#     hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

# python3 -m decalfc.pipeline.modeling --kwargs \
#     strategy="A --> G" \
#     trials=1 \
#     model-arch="Transformer" \
#     hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

python3 -m decalfc.pipeline.modeling --kwargs \
    strategy="A --> O" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.005, "scheduler": "plateau", "num_epochs": 200}'

python3 -m decalfc.pipeline.modeling --kwargs \
    strategy="E --> O" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.005, "scheduler": "plateau", "num_epochs": 200}'

python3 -m decalfc.pipeline.modeling --kwargs \
    strategy="G --> O" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.005, "scheduler": "plateau", "num_epochs": 200}'

python3 -m decalfc.pipeline.modeling --kwargs \
    strategy="A+E+G --> O" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'


python3 -m decalfc.pipeline.modeling --kwargs \
    strategy="O-2-2^ --> O-2-2^^" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

python3 -m decalfc.pipeline.modeling --kwargs \
    strategy="O --> A" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'


python3 -m decalfc.pipeline.modeling --kwargs \
    strategy="O --> E" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

python3 -m decalfc.pipeline.modeling --kwargs \
    strategy="O --> G" \
    trials=1 \
    model-arch="BLSTM" \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 200}'

