import os
import numpy as np
import torch
import torch.nn as nn
import mne 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AUTOENCODER_PATH = "supervised_1d_autoencoder.pth"
TRANSFORMER_PATH = "best_model_aug.pth"
INDICES_PATH = "pso_indices.pt"          #change name of the Feature selection algorithm in use
CONTROL_DIR = "Control_cleaned"
ADHD_DIR = "ADHD_cleaned"
Auto_DIM = 64  
DROPOUT = 0.2


class EEGAutoencoder(nn.Module):
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

class EEGTransformer(nn.Module):
    def __init__(self, input_dim, emb_dim=128, n_heads=None, n_layers=2, dropout=DROPOUT):
        super().__init__()
        self.input_dim = input_dim

        if n_heads is None:
            for h in [8, 6, 4, 2, 1]:
                if input_dim % h == 0:
                    n_heads = h
                    break
            else:
                raise ValueError(f"No valid number of heads for input_dim={input_dim}")

        self.pos = nn.Parameter(torch.randn(1, 1, input_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x + self.pos
        x = self.encoder(x)
        return self.classifier(x.squeeze(1))


def normalize_epochs(data):
    return (data - data.mean(axis=2, keepdims=True)) / (data.std(axis=2, keepdims=True) + 1e-6)

def plot_topomap(importance, title="Discriminative EEG Channel Map (ADHD vs Control)"):
    import mne
    import numpy as np
    import matplotlib.pyplot as plt

    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
    info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    
    data = np.expand_dims(importance, axis=1)
    evoked = mne.EvokedArray(data, info)

    
    fig = evoked.plot_topomap(
        times=0,
        time_format='',
        cmap='Reds',
        contours=5,
        show_names=True,  
        sphere=(0, 0.009, 0, 0.11),
        scalings=1,
        show=False
    )

    fig.suptitle(title, fontsize=12)
    plt.show()



def normalize_to_unit_range(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return np.zeros_like(arr)  
    norm = 2 * (arr - min_val) / (max_val - min_val) - 1  # Scale to [-1, 1]
    return norm



# ==== Load data ====
adhd_files = [os.path.join(ADHD_DIR, f) for f in os.listdir(ADHD_DIR) if f.endswith(".set")]
control_files = [os.path.join(CONTROL_DIR, f) for f in os.listdir(CONTROL_DIR) if f.endswith(".set")]
all_files = adhd_files + control_files
labels = [1] * len(adhd_files) + [0] * len(control_files)
_, val_files, _, val_labels = train_test_split(all_files, labels, test_size=0.2, stratify=labels, random_state=42)

# ==== Load models ====
autoencoder = EEGAutoencoder().to(DEVICE)
autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=DEVICE))
autoencoder.eval()

rsa_indices = torch.load(INDICES_PATH, weights_only=False)
rsa_indices = torch.tensor(rsa_indices).to(DEVICE)

transformer = EEGTransformer(input_dim=Auto_DIM).to(DEVICE)
transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=DEVICE))
transformer.eval()


print(" Extracting Biomarker Features...")
adhd_feats, control_feats = [], []

for fpath, label in zip(val_files, val_labels):
    data = mne.read_epochs_eeglab(fpath).get_data()
    data = normalize_epochs(data)
    data = torch.tensor(data, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        z = autoencoder.encode_only(data)  # (batch, 128)
        z_rsa = z[:, rsa_indices]
        pred = transformer(z_rsa).argmax(1).cpu().numpy()

    majority_pred = Counter(pred).most_common(1)[0][0]
    agreement_ratio = np.mean(pred == majority_pred)

    print(f"Pred: {majority_pred} | True: {label} | Agreement: {agreement_ratio:.2f}")

    if label == 1:
        adhd_feats.append(z_rsa.cpu())
    else:
        control_feats.append(z_rsa.cpu())

# ==== Channel Importance ====
print("\n Computing Channel Importance...")
if len(adhd_feats) == 0 or len(control_feats) == 0:
    raise ValueError("No features found! Check model predictions or class distribution.")

adhd_mean = torch.cat(adhd_feats).mean(dim=0)
control_mean = torch.cat(control_feats).mean(dim=0)
diff = (adhd_mean - control_mean).abs().numpy()

channel_importance = np.zeros(19)
features_per_ch = 128 // 19
rsa_indices_np = rsa_indices.cpu().numpy()

for ch in range(19):
    ch_indices = list(range(ch * features_per_ch, (ch + 1) * features_per_ch))
    rsa_ch_indices = [i for i, idx in enumerate(rsa_indices_np) if idx in ch_indices]
    if rsa_ch_indices:
        channel_importance[ch] = np.mean(diff[rsa_ch_indices])

np.save("pso_importance.npy", channel_importance)              #change name of the Feature selection algorithm in use
channel_importance = normalize_to_unit_range(channel_importance)
np.save("pso_importance_norm.npy", channel_importance)                #change name of the Feature selection algorithm in use
plot_topomap(channel_importance, "Discriminative EEG Channel Map (ADHD vs Control)")
