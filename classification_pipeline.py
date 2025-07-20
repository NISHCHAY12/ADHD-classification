import json
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mne
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time


# ==== Config ====
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15
DROPOUT = 0.2
LATENT_DIM = 128
Sel_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ==== RSA ====
def rsa_feature_selection(features, num_features_to_select=64, num_iters=50):
    best_score, best_subset = 0, None
    for _ in range(num_iters):
        idx = sorted(random.sample(
            range(features.shape[1]), num_features_to_select))
        subset = features[:, idx]
        score = (subset.std(dim=1).mean() - subset.mean(dim=1).std()).item()
        if score > best_score:
            best_score, best_subset = score, idx
    return best_subset


# ======= GA ==========
def ga_feature_selection(features, labels, num_features=64, num_generations=100, population_size=60):
    from geneticalgorithm import geneticalgorithm as ga
    ga.plt = None
    ga.plt_show = lambda *args, **kwargs: None

    def fitness(individual):
        idx = np.where(individual > 0.5)[0]
        if len(idx) == 0:
            return 1e6
        subset = features[:, idx]
        score = (subset.std(axis=1).mean() - subset.mean(axis=1).std())
        return -score

    dim = features.shape[1]
    varbound = np.array([[0, 1]] * dim)
    algorithm_param = {
        'max_num_iteration': num_generations,
        'population_size': population_size,
        'parents_portion': 0.3,
        'crossover_type': 'one_point',
        'mutation_probability': 0.1,
        'elit_ratio': 0.05,
        'crossover_probability': 0.5,
        'max_iteration_without_improv': 30
    }

    model = ga(function=fitness, dimension=dim, variable_type='bool', variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)
    model.run()
    best = model.output_dict['variable']

    # Force exactly 64 features
    scores = best
    sorted_indices = np.argsort(-scores)
    top_k = sorted_indices[:num_features]
    return sorted(top_k)


# ====== PSO =========
def pso_feature_selection(features, num_features=64):
    from pyswarm import pso

    def fitness(x):
        idx = np.where(x > 0.5)[0]
        if len(idx) == 0:
            return 1e6
        subset = features[:, idx]
        score = (subset.std(axis=1).mean() - subset.mean(axis=1).std())
        return -score

    lb = [0] * features.shape[1]
    ub = [1] * features.shape[1]
    xopt, _ = pso(fitness, lb, ub, swarmsize=40, maxiter=20)

    # Force exactly 64 features
    scores = xopt
    sorted_indices = np.argsort(-scores)
    top_k = sorted_indices[:num_features]
    return sorted(top_k)


# ====== RFE ========


def rfe_feature_selection(features, labels, num_features=64):
    clf = SVC(kernel="linear")
    rfecv = RFECV(
        estimator=clf,
        step=8,
        min_features_to_select=1,
        cv=StratifiedKFold(3),
        scoring='accuracy',
        n_jobs=-1
    )
    rfecv.fit(features, labels)

    ranking = rfecv.ranking_
    top_k = np.argsort(ranking)[:num_features]
    return sorted(top_k)


def augment(x):
    if torch.rand(1) < 0.5:
        x += torch.randn_like(x) * 0.05
    if torch.rand(1) < 0.3:
        t = np.random.randint(0, x.shape[1] - 10)
        x[:, t:t+10] = 0
    if torch.rand(1) < 0.3:
        ch = np.random.randint(0, x.shape[0])
        x[ch] = 0
    return x


class FeatureDataset(Dataset):
    def __init__(self, files, labels, encoder, idxs, augment_data=False):
        self.X, self.y = [], []
        self.augment = augment_data
        for f, label in zip(files, labels):
            x = mne.read_epochs_eeglab(f).get_data()
            x = (x - x.mean(axis=2, keepdims=True)) / \
                (x.std(axis=2, keepdims=True) + 1e-6)
            x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
            if self.augment:
                x = torch.stack([augment(e) for e in x])
            with torch.no_grad():
                z = encoder(x)[:, idxs].cpu()
            self.X.append(z)
            self.y.extend([label] * z.size(0))
        self.X = torch.cat(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EEGTransformer(nn.Module):
    def __init__(self, input_dim, emb_dim=128, n_heads=8, n_layers=2, dropout=DROPOUT):
        super().__init__()
        self.input_dim = input_dim

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


def get_class_weights(labels):
    counter = Counter(labels)
    total = sum(counter.values())
    weights = [total / counter[i] for i in sorted(counter.keys())]
    return torch.tensor(weights, dtype=torch.float32)


def train_model(model, train_loader, val_loader, class_weights):
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val, patience = 0, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total, correct = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_acc = 100 * correct / total
        val_acc = eval_acc(model, val_loader)
        print(
            f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

        if val_acc > best_val:
            best_val, patience = val_acc, 0
            torch.save(model.state_dict(), "best_model_aug.pth")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model_aug.pth"))
    evaluate(model, val_loader)


def eval_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    return 100 * correct / total


def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(1)
            y_true += yb.cpu().tolist()
            y_pred += preds.cpu().tolist()

    print("\n--- Evaluation Report ---")
    print(classification_report(y_true, y_pred,
          target_names=["Control", "ADHD"]))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
                "Control", "ADHD"], yticklabels=["Control", "ADHD"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    report = classification_report(y_true, y_pred, target_names=[
                                   "Control", "ADHD"], output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    classes = ["Control", "ADHD"]

    data = {metric: [report[cls][metric]
                     for cls in classes] for metric in metrics}
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 5))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, data[metric], width=width, label=metric)

    plt.xticks(x + width, classes)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Classification Report (Bar Chart)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_subject_level(model, files, labels, encoder, idxs):
    model.eval()
    preds, truths = [], []

    for fpath, true_label in zip(files, labels):
        x = mne.read_epochs_eeglab(fpath).get_data()
        x = (x - x.mean(axis=2, keepdims=True)) / \
            (x.std(axis=2, keepdims=True) + 1e-6)
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            features = encoder(x)[:, idxs]
            outputs = model(features)
            epoch_preds = outputs.argmax(1).cpu().numpy()

        majority_vote = Counter(epoch_preds).most_common(1)[0][0]
        preds.append(majority_vote)
        truths.append(true_label)

    print("\n=== Subject-Level Evaluation ===")
    print(f"Accuracy: {accuracy_score(truths, preds)*100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(truths, preds))
    print("Classification Report:")
    print(classification_report(truths, preds,
          target_names=["Control", "ADHD"]))

    cm = confusion_matrix(truths, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=[
                "Control", "ADHD"], yticklabels=["Control", "ADHD"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Subject-Level Confusion Matrix")
    plt.tight_layout()
    plt.show()
    return accuracy_score(truths, preds)*100


def serialize_report(report_dict):
    """Recursively convert any NumPy types to native Python types for JSON compatibility."""
    if isinstance(report_dict, dict):
        return {k: serialize_report(v) for k, v in report_dict.items()}
    elif isinstance(report_dict, np.ndarray):
        return report_dict.tolist()
    elif isinstance(report_dict, (np.float32, np.float64, np.int64)):
        return report_dict.item()
    else:
        return report_dict


def main():
    adhd = [
        f"ADHD_cleaned/{f}" for f in os.listdir("ADHD_cleaned") if f.endswith(".set")]
    ctrl = [
        f"Control_cleaned/{f}" for f in os.listdir("Control_cleaned") if f.endswith(".set")]
    files = adhd + ctrl
    labels = [1]*len(adhd) + [0]*len(ctrl)
    import matplotlib.pyplot as plt

    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    for method in ["rsa", "ga", "pso", "rfe"]:
        fold_accuracies = []
        fold_subject_accuracies = []
        fold_times = []

        print(f"\n==== Running 10-Fold CV with {method.upper()} ====")

        for fold, (train_idx, val_idx) in enumerate(kf.split(files, labels)):
            print(f"\n--- Fold {fold + 1} ---")
            train_f = [files[i] for i in train_idx]
            val_f = [files[i] for i in val_idx]
            train_y = [labels[i] for i in train_idx]
            val_y = [labels[i] for i in val_idx]

            supervised_model = SupervisedEEGAutoencoder()
            supervised_model.load_state_dict(torch.load(
                "supervised_1d_autoencoder.pth", map_location=DEVICE))
            supervised_model.to(DEVICE)
            supervised_model.eval()

            ae = supervised_model

            feats = []
            for f, label in zip(train_f, train_y):
                x = mne.read_epochs_eeglab(f).get_data()
                x = (x - x.mean(axis=2, keepdims=True)) / \
                    (x.std(axis=2, keepdims=True) + 1e-6)
                x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    z = ae.encode_only(x).cpu()
                feats.append(z)
            feats = torch.cat(feats)
            start = time.time()

            # 2. Feature selection
            if method == 'rsa':
                idxs = rsa_feature_selection(
                    feats, num_features_to_select=Sel_DIM)
                torch.save(idxs, f"rsa_indices_{fold + 1}.pt")
            elif method == 'ga':
                idxs = ga_feature_selection(
                    feats.numpy(), train_y, num_features=Sel_DIM)
                torch.save(idxs, f"ga_indices_{fold + 1}.pt")
            elif method == 'pso':
                idxs = pso_feature_selection(
                    feats.numpy(), num_features=Sel_DIM)
                torch.save(idxs, f"pso_indices_{fold + 1}.pt")
            elif method == 'rfe':
                feats_rfe = []
                expanded_labels = []
                for f, label in zip(train_f, train_y):
                    x = mne.read_epochs_eeglab(f).get_data()
                    x = (x - x.mean(axis=2, keepdims=True)) / \
                        (x.std(axis=2, keepdims=True) + 1e-6)
                    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        z = ae.encode_only(x).cpu()
                    feats_rfe.append(z)
                    expanded_labels.extend([label] * z.shape[0])
                feats_rfe = torch.cat(feats_rfe)
                idxs = rfe_feature_selection(
                    feats_rfe.numpy(), expanded_labels, num_features=Sel_DIM)
                torch.save(idxs, f"rfe_indices_{fold + 1}.pt")

            end = time.time()
            fold_times.append(end - start)
            train_ds = FeatureDataset(
                train_f, train_y, ae.encode_only, idxs, augment_data=True)
            val_ds = FeatureDataset(
                val_f, val_y, ae.encode_only, idxs, augment_data=False)
            train_dl = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

            model = EEGTransformer(input_dim=len(idxs)).to(DEVICE)
            weights = get_class_weights(train_ds.y.numpy())
            train_model(model, train_dl, val_dl, weights)

            epoch_acc = eval_acc(model, val_dl)
            subject_acc = evaluate_subject_level(
                model, val_f, val_y, ae.encode_only, idxs)

            fold_accuracies.append(epoch_acc)
            fold_subject_accuracies.append(subject_acc)

        results[method] = {
            "avg_epoch_accuracy": np.mean(fold_accuracies),
            "avg_subject_accuracy": np.mean(fold_subject_accuracies),
            "avg_fold_time": np.mean(fold_times),
            "all_epoch_accuracies": fold_accuracies,
            "all_subject_accuracies": fold_subject_accuracies,
            "all_fold_times": fold_times
        }

        with open("crossval_results64.json", "w") as f:
            json.dump(serialize_report(results), f, indent=4)


if __name__ == "__main__":
    main()
