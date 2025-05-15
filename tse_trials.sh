# generate trials
# python3 -m decalfc.pipeline.modeling --kwargs \
#     trial-type="tse" \
#     trials=3 \
#     hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 100, "hidden_size": 64, "num_layers": 2, "dropout_rate": 0.5}'

# summarize
python3 -m decalfc.abstractions.perfdata --kwargs \
    breakdown-type="tse"

