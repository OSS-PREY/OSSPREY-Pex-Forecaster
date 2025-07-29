import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    data_files = {
        "apache": "clean-apache-network-data-2-2.csv",
        "eclipse": "clean-eclipse-network-data-3-3.csv",
        "github": "clean-github-network-data-4-5.csv",
        "osgeo": "clean-osgeo-network-data-2-2.csv",
    }

    dfs = []
    for label, fname in data_files.items():
        df = pd.read_csv(f"Forecasting-Paper-Utils/{fname}")
        df = df.drop(columns=["proj_name"])  # drop the project name
        df["label"] = label
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def preprocess(df):
    X = df.drop(columns=["label"])
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
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.model(x)


def train_model(train_loader, val_loader, in_dim, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProjectRouterNet(in_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        # simple validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total else 0
        print(f"Epoch {epoch+1}/{epochs} - val acc: {acc:.3f}")
    return model


def prepare_loaders(X_train, X_val, y_train, y_val, batch_size=32):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def predict_project(model, scaler, sample_df):
    sample = scaler.transform(sample_df.drop(columns=["proj_name"]))
    with torch.no_grad():
        logits = model(torch.tensor(sample, dtype=torch.float32))
        pred = logits.argmax(dim=1).item()
    labels = ["apache", "eclipse", "github", "osgeo"]
    return labels[pred]


if __name__ == "__main__":
    df = load_data()
    (X_train, X_val, y_train, y_val), scaler = preprocess(df)
    train_loader, val_loader = prepare_loaders(X_train, X_val, y_train, y_val)
    model = train_model(train_loader, val_loader, X_train.shape[1])

    # demo on a single sample
    sample_project = df.iloc[[0]]  # first row from dataset
    foundation = predict_project(model, scaler, sample_project)
    print("Predicted foundation:", foundation)
