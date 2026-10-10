import argparse

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

torch.manual_seed(42)
parser = argparse.ArgumentParser()

parser.add_argument(
    "--components",
    type=int,
    default=25
)

args = parser.parse_args()

n_components = args.components


digits = load_digits()

# pyrefly: ignore [missing-attribute]
X = digits.data
# pyrefly: ignore [missing-attribute]
y = digits.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


print("Original shape:", X_train_scaled.shape)
print("PCA shape:", X_train_pca.shape)

print(
    "PCA explained variance:",
    pca.explained_variance_ratio_.sum()
)


def create_loaders(X_train, X_test, y_train, y_test):

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, test_loader


class DigitClassifier(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


def train_model(
    train_loader,
    test_loader,
    input_size,
    epochs=20
):

    model = DigitClassifier(input_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(epochs):

        model.train()

        for X_batch, y_batch in train_loader:

            outputs = model(X_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            outputs = model(X_batch)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    accuracy = correct / total

    return model, accuracy


original_train_loader, original_test_loader = create_loaders(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)


pca_train_loader, pca_test_loader = create_loaders(
    X_train_pca,
    X_test_pca,
    y_train,
    y_test
)


original_model, original_accuracy = train_model(
    original_train_loader,
    original_test_loader,
    input_size=64
)


pca_model, pca_accuracy = train_model(
    pca_train_loader,
    pca_test_loader,
    input_size=n_components
)


print("\nResults")
print("-" * 40)

print(
    f"Original (64 features): "
    f"{original_accuracy:.4f}"
)

print(
    f"PCA ({n_components} features): "
    f"{pca_accuracy:.4f}"
)

print(
    f"\nAccuracy difference: "
    f"{(original_accuracy - pca_accuracy):.4f}"
)