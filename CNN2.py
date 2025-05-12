import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import pandas as pd
import sys

class DualLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# redirect stdout
sys.stdout = DualLogger("output_log.txt")


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

            nn.Linear(128 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_model(train_dataset, val_dataset, patience=4, batch_size=64, epochs=100):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    torch.cuda.empty_cache()

    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

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
        scheduler.step(avg_val_loss)

        # clear cache
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


def evaluate_model(model, test_dataset, batch_size=8):
    model.eval()
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


def load_preprocess_images(csv_path, filter_40x=True):
    df = pd.read_csv(csv_path)
    if filter_40x:
        df = df[df['mag'] == 40]
    return df.reset_index(drop=True)


class ImageDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=10, scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]) if augment else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filename']).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(row['tumor_class'], dtype=torch.float32).unsqueeze(0)
        return img, label


def split_by_balanced_patient_assignment(df):
    # 1. get count of samples per patient
    patient_counts = df.groupby('patient_id')['tumor_class'].agg(['count', 'mean']).reset_index()

    # 2. split into two groups by label (benign = 0, malignant = 1)
    benign_df = patient_counts[patient_counts['mean'] == 0].sort_values(by='count', ascending=False)
    malignant_df = patient_counts[patient_counts['mean'] == 1].sort_values(by='count', ascending=False)

    # 3. initialize empty groups
    train_ids, val_ids, test_ids = set(), set(), set()

    # 4. round-robin assignment from benign patients
    for i, pid in enumerate(benign_df['patient_id']):
        if i % 5 in [0, 1, 2]:
            train_ids.add(pid)
        elif i % 5 == 3:
            val_ids.add(pid)
        else:
            test_ids.add(pid)

    # 5. round-robin assignment from malignant patients
    for i, pid in enumerate(malignant_df['patient_id']):
        if i % 5 in [0, 1, 2]:
            train_ids.add(pid)
        elif i % 5 == 3:
            val_ids.add(pid)
        else:
            test_ids.add(pid)

    # 6. create new dataframes by filtering original df
    df_train = df[df['patient_id'].isin(train_ids)].reset_index(drop=True)
    df_val = df[df['patient_id'].isin(val_ids)].reset_index(drop=True)
    df_test = df[df['patient_id'].isin(test_ids)].reset_index(drop=True)

    # 7. print distribution summary
    for name, subset in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        total = len(subset)
        class_counts = subset['tumor_class'].value_counts().to_dict()
        print(f"{name} set: {total} samples — class distribution: {class_counts}")

    return df_train, df_val, df_test


# load and preprocess data
df = load_preprocess_images("folds_updated.csv", filter_40x=False)
print("loaded image paths and labels")

# split and apply pca
df_train, df_val, df_test = split_by_balanced_patient_assignment(df)
train_dataset = ImageDataset(df_train, augment=True)
val_dataset = ImageDataset(df_val, augment=False)
test_dataset = ImageDataset(df_test, augment=False)
print("split data")

# train model
model = train_model(train_dataset, val_dataset)

# evaluate on test set
evaluate_model(model, test_dataset)
