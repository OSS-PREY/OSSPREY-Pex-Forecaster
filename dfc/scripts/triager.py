# %% [markdown]
# # Triage Results
# Script/NB for generating the lookup of the triage results. For every project in
# the test set of each of ASF, EF, OF, and GH we will predict on the best model of
# each. Later, we can run the triage model and pick the output label.
# 
# Output columns will be:
# - Incubator
# - Project
# - Outcome
# - ASF Pred
# - EF Pred
# - OF Pred
# - GF Pred
# - Triage Incubator Pred (to be done later)

# %% [markdown]
# ***

# %% [markdown]
# ## Env Setup

# %%
import decalfc as pex
from decalfc.abstractions.modeldata import *
from decalfc.abstractions.tsmodel import *
from decalfc.pipeline.inference import *

import numpy as np

# %%
incs = ["apache", "eclipse", "osgeo", "github"]
model_archs = dict(zip(incs, ["BLSTM", "BLSTM", "DLSTM", "DLSTM"]))

# %% [markdown]
# ***
# ## Finding the Median Fold Projects

# %%
def find_median_fold(inc: str, k: int=5, model_arch: str="BLSTM", **kwargs) -> tuple[set[str], set[str]]:
    """
    Finds the median fold from the cross-validation, i.e. the projects
    themselves that we used to generate the CV results reported in the paper.
    """
    
    # === train the models for this fold === #
    # load all data in a folds iterator
    folds = ModelData.gen_k_folds(
        transfer_strategy=f"{inc} --> {inc}",
        transform_kwargs=kwargs.get("transform-kwargs", dict()),
        nfolds=k,
        yield_projs=True
    )
    
    # store only the median trial in the perf database; we'll create a temporary
    # perf-db for this kfold trial, pick the median, and update the actual perf
    # db
    temp_perf_path = f""
    pfd = PerfData(temp_perf_path)
    fold_projs: list[tuple[set[str], set[str]]] = list()
    perfs = list()

    # train model & test for each fold
    for fold, projs in folds:
        ## grab sample tensor, i.e. first tensor we have
        sample_tensor = fold.tensors["train"]["x"][0]
            
        ## ensure some hyperparams
        hyperparams = {"input_size": sample_tensor.shape[1]}
        hyperparams.update(kwargs.get("hyperparams", dict()))

        ## build model
        model = TimeSeriesModel(
            model_arch=model_arch,
            hyperparams=hyperparams
        )
        
        ## train & test
        print(len(fold.tensors["train"]["y"]), len(fold.tensors["test"]["y"]))
        print(len(projs[0]), len(projs[1]))
        model.train(fold)
        model.test(fold)
        
        ## track perf
        pfd._add_entry(
            transfer_strat=f"{inc} --> {inc}",
            model_arch=model_arch,
            preds=model.preds,
            targets=model.targets,
            export_db=False
        )
        perfs.append(pfd.data[(pfd.data.metric == "f1-score") & (pfd.data.label == "weighted avg")]["perf"].iloc[-1])
        
        ## add projs
        fold_projs.append(projs)
    
    # median finding
    median_idx = np.where(perfs == np.median(perfs))[0][0]

    # export train and test for median fold
    return fold_projs[median_idx]

# %%
median_folds = {
    inc: find_median_fold(inc=inc[0].upper(), model_arch=model_archs[inc])
    for inc in incs
}

# %%
median_folds = {
    inc: (set(f[0]), set(f[1]))
    for inc, f in median_folds.items()
}

# %%
from json import dumps
print(list(map(len, median_folds["apache"])))
print(list(map(len, median_folds["eclipse"])))
print(list(map(len, median_folds["osgeo"])))
print(list(map(len, median_folds["github"])))

# %%
print(median_folds)

# %% [markdown]
# ***
# ## Inference on the Median Folds

# %%
def train_helper(m: TimeSeriesModel, X, y):
    # track losses
    losses = {}
    test_losses = {}
    best_loss = float("inf")
    best_epoch = 0
    patience = 10
    TOLERANCE = 1e-4
    
    # initialize optimizer and scheduler
    m.optimizer = torch.optim.AdamW(
        m.model.parameters(),
        lr=m.hyperparams["learning_rate"],
        weight_decay=0.01
    )
    m.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        m.optimizer, mode="min", factor=0.5, patience=10
    )

    m.loss_fc = FocalLoss(alpha=0.5, gamma=2.0)

    # Training loop
    for epoch in range(m.hyperparams["num_epochs"]):
        m.model.train()
        losses[epoch] = []

        # stochastic GD
        for data, target in tqdm(list(zip(X, y))):
            # shape into (B --> 1, ntimesteps, nfeatures)
            data = data.to(m.device).reshape(1, data.shape[0], -1)
            target = target.to(m.device).to(torch.float32)

            # forward pass
            pred = m.model.predict(data)[..., 1].to(torch.float32)

            # compute loss (Focal Loss)
            loss = m.loss_fc(pred, target)
            
            # backward pass
            m.optimizer.zero_grad()
            loss.backward()

            # grad clipping
            torch.nn.utils.clip_grad_norm_(m.model.parameters(), max_norm=1.0)

            # step the optimizer
            m.optimizer.step()

            # loss tracking
            losses[epoch].append(loss.item())

        # mean loss
        losses[epoch] = np.mean(losses[epoch])

        # logging
        current_lr = m.optimizer.param_groups[0]["lr"]
        log(f"Epoch [{epoch + 1}/{m.hyperparams['num_epochs']}] | "
            f"Loss: {losses[epoch]:.4f}, "
            f"LR: {current_lr:.6f}", "log")

        # lr scheduling
        if m.scheduler is not None:
            m.scheduler.step(losses[epoch])

        # early stopping protocol
        avg_loss = losses[epoch]

        if avg_loss < best_loss - TOLERANCE:
            best_loss = avg_loss
            best_model_weights = copy.deepcopy(m.model.state_dict())      
            patience = 10
            best_epoch = epoch
        else:
            patience -= 1
            if patience == 0:
                log("Early stopping triggered. Loading best model weights.", "log")
                m.model.load_state_dict(best_model_weights)
                break

    # check for divergence
    if np.isnan(sorted(losses.items(), reverse=True)[0][1]) or np.isinf(sorted(losses.items(), reverse=True)[0][1]):
        log("NaN or Inf loss generated, i.e. failed to converge: ignoring and exiting", "error")

    log("Training completed.", "log")

    print(f"Model Name: {m.model_arch}")
    print(f"Input size: {m.hyperparams['input_size']}")
    print(f"Hidden size: {m.hyperparams['hidden_size']}")
    print(f"Number of layers: {m.hyperparams['num_layers']}")
    print(f"Shape of X_train: {X[2].shape}")

# %%
def netdata_inference(nd: NetData, model: TimeSeriesModel, test_set: set[str]):
    """
    Inference on the test set of the NetData
    """
    
    # tracker
    predictions = dict()
    predictions["project"] = list()
    predictions["prediction"] = list()
    predictions["target"] = list()
    
    # eval
    model.model.eval()
    with torch.no_grad():
        # iterate projects
        for proj_name, X in nd.data_dict.items():
            # skip
            if proj_name not in test_set:
                continue
            
            # convert to tensor
            X = torch.tensor(X).to(model.device)
            X = X.reshape(1, X.shape[0], -1)
            y_true = (
                "graduated" if proj_name in nd.project_status["graduated"] else
                ("retired" if proj_name in nd.project_status["retired"] else "incubating")
            )
            
            # predictions
            out = model.model(X)
            pred_label = torch.argmax(out, dim=1)
            y_pred = pred_label.cpu().numpy()[0]
            y_pred = "graduated" if y_pred == 1 else "retired"
            
            # update trackers
            predictions["project"].append(proj_name)
            predictions["prediction"].append(y_pred)
            predictions["target"].append(y_true)
    
    # export
    return predictions

def fold_inference(inc: str, folds: dict[str, tuple[set[str], set[str]]], model_arch: str="BLSTM", **kwargs):
    """
    Trains a model on the given train projects and outputs labels for all test
    projects.
    """
    
    # build the train and test sets for all incubators
    nds = {
        i: NetData(
            incubator=i,
            split_set={"train": folds[i][0], "test": folds[i][1]},
            is_train="both",
            verbose=False,
        )
        for i in incs
    }
    
    # setup model & train on the train set
    model = TimeSeriesModel(
        model_arch=model_arch,
        **kwargs
    )
    train_helper(
        model,
        X=nds[inc].tensors["train"]["x"],
        y=nds[inc].tensors["train"]["y"],
    )
    
    # inference on each project in the nd
    return {
        i: netdata_inference(nds[i], model, test_set=folds[i][1])
        for i in incs
    }

# %%
# grab pred results
res = {
    inc: fold_inference(inc, median_folds, model_archs[inc])
    for inc in incs
}

# %%
import pandas as pd

res = {
    inc: {
        i: pd.DataFrame(res[inc][i])
        for i in res[inc]
    }
    for inc in incs
}
res

# %% [markdown]
# ***
# ## Organizing Results

# %%
def transform_inc_res(inc_res, inc: str):
    for i, df in inc_res.items():
        df["incubator"] = i
        df.rename(columns={"prediction": f"{inc}_pred"}, inplace=True)
        
    return pd.concat(inc_res.values(), ignore_index=True)

# %%
r1 = [transform_inc_res(r, i) for i, r in res.items()]
r2 = pd.concat(r1, axis=1, join="outer")
r3 = r2.loc[:, ~r2.columns.duplicated()]
r3

# %%
df = r3[["incubator", "project", "target", "apache_pred", "eclipse_pred", "osgeo_pred", "github_pred"]]
df.to_csv("triage_results.csv", index=False)

# %%
df

# %%
f = {i: (list(f[0]), list(f[1])) for i, f in median_folds.items()}
f

# %%
import json

with open("triage_median_folds.json", "w") as file:
    json.dump(f, file, indent=4)


