import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(drop_proj_name: bool = True):
    data_files = {
        "apache": "clean-apache-network-data-2-2.csv",
        "eclipse": "clean-eclipse-network-data-3-3.csv",
        "github": "clean-github-network-data-4-5.csv",
        "osgeo": "clean-osgeo-network-data-2-2.csv",
    }

    dfs = []
    for label, fname in data_files.items():
        df = pd.read_csv(f"Forecasting-Paper-Utils/{fname}")
        if drop_proj_name:
            df = df.drop(columns=["proj_name"])  # drop the project name
        df["label"] = label
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def preprocess(df, drop_proj_name: bool = True):
    cols_to_drop = ["label"]
    if drop_proj_name and "proj_name" in df.columns:
        cols_to_drop.append("proj_name")
    X = df.drop(columns=cols_to_drop)
    y = df["label"].factorize()[0]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler


class ProjectRouterNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectRouterNet(in_dim).to(device)
    criterion = FocalLoss(alpha=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
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


def predict_project(model, scaler, sample_df):
    cols_to_drop = ["proj_name"]
    if "label" in sample_df.columns:
        cols_to_drop.append("label")
    sample = scaler.transform(sample_df.drop(columns=cols_to_drop))
    with torch.no_grad():
        logits = model(torch.tensor(sample, dtype=torch.float32))
        pred = logits.argmax(dim=1).item()
    labels = ["apache", "eclipse", "github", "osgeo"]
    return labels[pred]


if __name__ == "__main__":
    df = load_data()
    (X_train, X_val, y_train, y_val), scaler = preprocess(df)
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

    # demo on a single sample using data with project names
    df_with_names = load_data(drop_proj_name=False)
    sample_project = df_with_names.iloc[[0]]
    foundation = predict_project(model, scaler, sample_project)
    print("Predicted foundation:", foundation)

    # run prediction for all projects and save results
    predictions = []
    for _, row in df_with_names.iterrows():
        pred = predict_project(model, scaler, row.to_frame().T)
        predictions.append({
            "project-name": row["proj_name"],
            "target-foundation": row["label"],
            "predicted-foundation": pred,
        })

    out_df = pd.DataFrame(predictions)
    out_df.to_csv("project-router-output.csv", index=False)
