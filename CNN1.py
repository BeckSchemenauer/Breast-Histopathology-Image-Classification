from preprocessing import load_preprocess_images, split_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# check if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_model(X_train, y_train, X_val, y_val, patience=5, batch_size=4, epochs=100):
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = torch.tensor(self.X[idx], dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
            y = torch.tensor([self.y[idx]], dtype=torch.float32)
            return x, y

    train_dataset = ImageDataset(X_train, y_train)
    val_dataset = ImageDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # track loss and accuracy
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # loop through input (xb) and target batches (yb)
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (preds >= 0.5).float()
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(correct / total)

        # evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()
                predicted = (outputs >= 0.5).float()
                val_correct += (predicted == yb).sum().item()
                val_total += yb.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"epoch {epoch + 1}, val loss: {avg_val_loss:.4f}, val acc: {val_acc:.4f}")

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            # if there has been no improvement for a sufficient amount of time, stop
            if patience_counter >= patience:
                print("early stopping triggered.")
                break

        # optionally clear cache
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    model.load_state_dict(best_model_state)

    # plot training/validation curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    return model


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



def evaluate_model(model, X_test, y_test, batch_size=8):
    model.eval()

    # convert to tensors and permute to [N, C, H, W]
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    preds_all = []
    y_true_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            probs = model(xb).squeeze().cpu()
            preds = (probs >= 0.5).int().numpy()
            y_true = yb.squeeze().cpu().int().numpy()

            preds_all.extend(preds)
            y_true_all.extend(y_true)

    print("classification report:")
    print(classification_report(y_true_all, preds_all))

    cm = confusion_matrix(y_true_all, preds_all)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("confusion matrix")
    plt.show()

    acc = accuracy_score(y_true_all, preds_all)
    print(f"final test accuracy: {acc:.4f}")


# load and preprocess data
X, y, ids = load_preprocess_images("folds_updated.csv", filter_40x=True)
print("loaded and preprocessed data")

# split and apply pca
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
print("split data")

# report sizes of each split
print(f"train size: {X_train.shape[0]} samples")
print(f"validation size: {X_val.shape[0]} samples")
print(f"test size: {X_test.shape[0]} samples")

# report class balances in each split
for split_name, labels in [("train", y_train), ("validation", y_val), ("test", y_test)]:
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique.astype(int), counts))
    print(f"{split_name} class distribution: {dist}")

# train model
model = train_model(X_train, y_train, X_val, y_val)

# evaluate on test set
evaluate_model(model, X_test, y_test)
