import os
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mne
from sklearn.model_selection import train_test_split

# ==== Configuration ====
EPOCH_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SupervisedEEGDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.X, self.y = [], []
        for path, label in zip(file_paths, labels):
            epochs = mne.read_epochs_eeglab(path).get_data()
            epochs = (epochs - epochs.mean(axis=2, keepdims=True)) / (epochs.std(axis=2, keepdims=True) + 1e-6)
            for epoch in epochs:
                self.X.append(torch.tensor(epoch, dtype=torch.float32))
                self.y.append(label)
        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]



class SupervisedEEGAutoencoder(nn.Module):
    def __init__(self, input_channels=19, time_steps=256, latent_dim=128, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.latent_fc = nn.Linear(32 * time_steps, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * time_steps),
            nn.Unflatten(1, (32, time_steps)),
            nn.ConvTranspose1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_channels, kernel_size=3, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        z = self.latent_fc(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return logits, recon, z

    def encode_only(self, x):
        x = self.encoder(x)
        return self.latent_fc(x)



def train_supervised_autoencoder(model, loader, class_weights=None):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE) if class_weights is not None else None)
    rec_loss_fn = nn.MSELoss()

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_cls, total_rec = 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits, recon, _ = model(xb)
            loss_cls = cls_loss_fn(logits, yb)
            loss_rec = rec_loss_fn(recon, xb)
            loss = loss_cls + 0.5 * loss_rec

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_cls += loss_cls.item()
            total_rec += loss_rec.item()

        print(f"Epoch {epoch+1}: Cls Loss = {total_cls:.4f}, Recon Loss = {total_rec:.4f}")



def main():
    adhd_files = [os.path.join("ADHD_cleaned", f) for f in os.listdir("ADHD_cleaned") if f.endswith(".set")]
    ctrl_files = [os.path.join("Control_cleaned", f) for f in os.listdir("Control_cleaned") if f.endswith(".set")]
    files = adhd_files + ctrl_files
    labels = [1]*len(adhd_files) + [0]*len(ctrl_files)

    train_files, _, train_labels, _ = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

    dataset = SupervisedEEGDataset(train_files, train_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SupervisedEEGAutoencoder()
    class_weights = torch.tensor([1.0, 1.0])  
    train_supervised_autoencoder(model, loader, class_weights=class_weights)

    torch.save(model.state_dict(), "supervised_1d_autoencoder.pth")
    visualize_tsne(model, dataset)



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_tsne(autoencoder, dataset, n_samples=500):
    autoencoder.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    embeddings = []
    labels = []

    count = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        with torch.no_grad():
            z = autoencoder.encode_only(xb).cpu().numpy()
        embeddings.append(z)
        labels.extend(yb.tolist())
        count += z.shape[0]
        if count >= n_samples:
            break

    embeddings = np.vstack(embeddings)[:n_samples]
    labels = labels[:n_samples]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    label_names = ["Control", "ADHD"]
    plt.figure(figsize=(8, 6))
    for label in [0, 1]:
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label_names[label], alpha=0.7)
    plt.title("t-SNE of Autoencoder Latent Space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()                       