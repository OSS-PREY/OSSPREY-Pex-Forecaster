import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    """Load all foundation datasets with project names and labels."""

    data_files = {
        "apache": "clean-apache-network-data-2-2.csv",
        "eclipse": "clean-eclipse-network-data-3-3.csv",
        "github": "clean-github-network-data-4-5.csv",
        "osgeo": "clean-osgeo-network-data-2-2.csv",
    }

    dfs = []
    for label, fname in data_files.items():
        df = pd.read_csv(f"Forecasting-Paper-Utils/{fname}")
        df["label"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def preprocess(df):
    """Scale features and encode labels while preserving project names."""

    project_names = df["proj_name"].to_numpy()
    labels_str = df["label"].to_numpy()
    y, label_names = pd.factorize(labels_str)

    X = df.drop(columns=["proj_name", "label"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, project_names, labels_str, scaler, label_names


class ProjectRouterNet(nn.Module):
    """Deeper network with batch norm and dropout for richer representations."""

    def __init__(self, in_dim, out_dim: int = 4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.model(x)


class FocalLoss(nn.Module):
    """Focal loss for tackling class imbalance."""

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def train_model(
    train_loader,
    val_loader,
    in_dim,
    class_weights,
    epochs: int = 200,
    patience: int = 20,
):
    """Train the model with early stopping and adaptive learning rate.

    Args:
        train_loader: Data loader for training data.
        val_loader: Data loader for validation data.
        in_dim: Number of input features.
        class_weights: Tensor of class weights for the focal loss.
        epochs: Maximum number of training epochs.
        patience: Number of epochs to wait for improvement before stopping.
    """

    model = ProjectRouterNet(in_dim).to(DEVICE)
    criterion = FocalLoss(alpha=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * yb.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_loss /= total if total else 1
        acc = correct / total if total else 0
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} - val acc: {acc:.3f} - val loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    return model


def evaluate(model, loader):
    """Compute accuracy on a data loader."""

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total if total else 0


def prepare_loaders(X_train, X_val, y_train, y_val, batch_size=32):
    train_y = torch.tensor(y_train, dtype=torch.long)
    class_counts = torch.bincount(train_y)
    class_weights = (1.0 / class_counts.float())
    sample_weights = class_weights[train_y]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), train_y)
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader, class_weights


def predict_project(model, scaler, sample_df, label_names=None):
    cols_to_drop = ["proj_name"]
    if "label" in sample_df.columns:
        cols_to_drop.append("label")
    sample = scaler.transform(sample_df.drop(columns=cols_to_drop))
    with torch.no_grad():
        logits = model(torch.tensor(sample, dtype=torch.float32).to(DEVICE))
        pred = logits.argmax(dim=1).item()
    if label_names is None:
        label_names = ["apache", "eclipse", "github", "osgeo"]
    return label_names[pred]


if __name__ == "__main__":
    set_seed(42)
    df = load_data()
    X, y, proj_names, labels_str, scaler, label_names = preprocess(df)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    predictions = []
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_loader, val_loader, class_weights = prepare_loaders(
            X_train, X_val, y_train, y_val
        )
        model = train_model(
            train_loader,
            val_loader,
            X_train.shape[1],
            class_weights,
            epochs=200,
            patience=20,
        )
        acc = evaluate(model, val_loader)
        accuracies.append(acc)
        print(f"Fold {fold} accuracy: {acc:.3f}")

        fold_preds = []
        model.eval()
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                fold_preds.extend(out.argmax(1).cpu().numpy())
        for idx, pred in zip(val_idx, fold_preds):
            predictions.append(
                {
                    "project-name": proj_names[idx],
                    "target-foundation": labels_str[idx],
                    "predicted-foundation": label_names[pred],
                }
            )

    print(f"Average CV accuracy: {np.mean(accuracies):.3f}")
    pd.DataFrame(predictions).to_csv("project-router-output.csv", index=False)

    sample = predictions[0]
    print(
        "Sample project:",
        sample["project-name"],
        "target:",
        sample["target-foundation"],
        "predicted:",
        sample["predicted-foundation"],
    )
