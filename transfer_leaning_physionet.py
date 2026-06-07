import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

seed = 128
random.seed(seed)
# Python environment
os.environ['PYTHONHASHSEED'] = str(seed)
# Numpy
np.random.seed(seed)
# PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# PyTorch CUDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# 1. SOTA MODEL DEFINITION (EEGNet)
# ==========================================
class TinyEEGNet(nn.Module):
    def __init__(self, nb_classes=2, Chans=29, Samples=2001, dropoutRate=0.65, kernelLength=125, F1=4, D=2, F2=8):
        super(TinyEEGNet, self).__init__()

        # Block 1 - Temporal Convolution
        # kernelLength reduced to 125 (0.25s at 500Hz) to shrink parameter footprint
        self.block1_temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernelLength), padding=(0, kernelLength // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        # Block 1 - Spatial Depthwise Convolution
        self.block1_spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            # AGGRESSIVE POOLING: Increased from (1, 4) to (1, 8)
            nn.AvgPool2d((1, 8)),
            # SPATIAL DROPOUT: Drops entire 2D feature maps instead of individual elements
            nn.Dropout2d(dropoutRate)
        )

        # Block 2 - Separable Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            # AGGRESSIVE POOLING: Increased from (1, 8) to (1, 16)
            nn.AvgPool2d((1, 16)),
            nn.Dropout2d(dropoutRate)
        )

        # Dynamically calculate flattening features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            dummy = self.block2(self.block1_spatial(self.block1_temporal(dummy)))
            self.flatten_size = dummy.view(1, -1).size(1)

        self.classifier = nn.Linear(self.flatten_size, nb_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add single channel dimension [Batch, 1, Chans, Samples]
        x = self.block1_temporal(x)
        x = self.block1_spatial(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
# ==========================================
# 2. DATA ACQUISITION & PREPROCESSING
# ==========================================
def load_and_preprocess_source(subjects=[1, 2, 3, 4, 5], sfreq_target=500):
    """
    Downloads PhysioNet MI data, matches it against standard target channels,
    and upsamples to 500 Hz to align with user specifications.
    """
    runs = [4, 8, 12]  # Left vs Right Hand Motor Imagery tasks
    raw_files = []
    for sub in subjects:
        raw_files.extend(eegbci.load_data(sub, runs))

    raws = [read_raw_edf(f, preload=True) for f in raw_files]
    raw = concatenate_raws(raws)

    # Standardize channel names (strip trailing dots from PhysioNet labels)
    raw.rename_channels(lambda x: x.strip('.'))

    # Filter using common cortical channels present in your 33-channel layout
    target_ch_names = [
        'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4',
        'FT7', 'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CPz', 'CP3', 'CP4',
        'TP7', 'TP8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2'
    ]
    available_channels = [ch for ch in target_ch_names if ch in raw.ch_names]
    raw.pick(available_channels)
    # raw.filter(l_freq=6.0, h_freq=10.0, method='iir', iir_params=dict(order=3, ftype='butter'), verbose=False)

    # Match user sampling frequency
    raw.resample(sfreq_target)

    # Extract structural Left (T1) and Right (T2) events
    events, event_id = mne.events_from_annotations(raw, regexp='T1|T2')

    # Epoching: 0 to 4 seconds post-stimulus
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=4.0,
                        baseline=None, preload=True, verbose=False)

    X = epochs.get_data(copy=True).astype(np.float32)
    y = (epochs.events[:, -1] - min(epochs.events[:, -1])).astype(np.int64)  # Normalize to 0 and 1

    return X, y, available_channels

# ==========================================
# 3. TRAINING & EVALUATION (10-FOLD CV)
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    X, y, common_channels = load_and_preprocess_source(subjects=list(range(1,30)))
    num_channels, num_samples = X.shape[1], X.shape[2]
    print(f"Dataset Loaded. Shape: {X.shape}, Clean Matching Channels: {len(common_channels)}")

    # Initialize Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = torch.tensor(X[train_idx]), torch.tensor(X[val_idx])
        y_train, y_val = torch.tensor(y[train_idx]), torch.tensor(y[val_idx])

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16, shuffle=False)

        # Instantiate SOTA EEGNet model
        model = TinyEEGNet(nb_classes=2, Chans=num_channels, Samples=num_samples).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        # Training loop per fold
        for epoch in range(90):  # Adjust epochs depending on your cluster capability
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        # Validation evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        acc = (correct / total) * 100
        fold_accuracies.append(acc)
        print(f"Fold {fold+1}/10 Accuracy: {acc:.2f}%")

    print(f"\nMean 10-Fold CV Accuracy: {np.mean(fold_accuracies):.2f}%")

    # Save base pre-trained model for fine-tuning
    torch.save({
        'model_state_dict': model.state_dict(),
        'channels': common_channels,
        'sfreq': 500
    }, 'eegnet_pretrained_base.pt')
    print("Pre-trained model saved successfully.")
