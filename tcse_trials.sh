python3 -m decalfc.pipeline.modeling --kwargs \
    trial-type="tcse" \
    trials=1 \
    hyperparams='{"learning_rate": 0.001, "scheduler": "plateau", "num_epochs": 100, "hidden_size": 64, "num_layers": 2, "dropout_rate": 0.5}'
