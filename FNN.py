from preprocessing import load_preprocess_images, split_and_apply_pca
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_model(X_train, y_train, X_val, y_val, patience=5, batch_size=32, epochs=100):
    # convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # create datasets and loaders for train and val
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # create model from custom class, set loss function and optimizer
    model = FNN()
    criterion = nn.BCELoss() # binary cross entropy
    optimizer = optim.Adam(model.parameters())

    # set up patience counter to determine early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        # loop through input (xb) and target batches (yb)
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_losses = [criterion(model(xb), yb) for xb, yb in val_loader]
            avg_val_loss = torch.mean(torch.stack(val_losses)).item()

        print(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}")

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            # if there has been no improvement for a sufficient amount of time, stop
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # load best model
    model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # predict probabilities and round to 0/1
        probs = model(X_test_tensor).squeeze()
        preds = (probs >= 0.5).int().numpy()
        y_true = y_test_tensor.int().numpy()

        # classification report
        print("Classification Report:")
        print(classification_report(y_true, preds))

        # confusion matrix
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        # final test accuracy
        acc = accuracy_score(y_true, preds)
        print(f"Final Test Accuracy: {acc:.4f}")


X, y = load_preprocess_images("folds_updated.csv")

X_train, X_val, X_test, y_train, y_val, y_test, pca = split_and_apply_pca(X, y)

model = train_model(X_train, y_train, X_val, y_val)

evaluate_model(model, X_test, y_test)

