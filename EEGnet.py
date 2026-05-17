from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
import numpy as np

from interfaces import BaseClassifier


class EEGNetModule(nn.Module):
    def __init__(self, nb_classes, chans=64, samples=128, dropout_rate=0.5,
                 kern_length=64, f1=8, f2=16, d=2):
        super(EEGNetModule, self).__init__()

        # Block 1: Temporal Convolution + Depthwise Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            # Depthwise Conv
            nn.Conv2d(f1, f1 * d, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Block 2: Separable Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Calculate Flatten size
        # Default for 128 samples: 128 -> (Pool 4) 32 -> (Pool 8) 4
        flatten_size = f2 * (samples // 32)

        self.classifier = nn.Linear(flatten_size, nb_classes)

    def forward(self, x):
        # Input shape: (Batch, Channels, Samples)
        x = x.unsqueeze(1)  # Add "color" channel: (Batch, 1, Channels, Samples)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EEGNetModuleMinimized(nn.Module):
    def __init__(self, nb_classes, chans=64, samples=128, f1=8, f2=16, d=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
            # Block 2
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5),
            nn.Flatten()
        )
        self.classifier = nn.Linear(f2 * (samples // 32), nb_classes)
    def forward(self, x):
        # Input shape: (Batch, Channels, Samples)
        x = x.unsqueeze(1)  # Add "color" channel: (Batch, 1, Channels, Samples)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EEGNet_CNN_LSTM_TCN_Concat(nn.Module):
    def __init__(self, nb_classes, chans=64, samples=128, f1=8, f2=16, d=2, lstm_hidden=16, tcn_channels=16):
        super().__init__()

        # 1. CNN Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5),

            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5)
        )

        # 2. GRU Temporal Extractor (Bidirectional)
        self.lstm = nn.GRU(input_size=f2, hidden_size=lstm_hidden // 2,
                           num_layers=1, batch_first=True, bidirectional=True)

        # 3. TCN Temporal Extractor (1D Dilated Convolution)
        # Kernel=3, dilation=2 adds a larger temporal receptive field
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=f2, out_channels=tcn_channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(tcn_channels),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1) # Pool across time to get a fixed size vector per batch
        )

        # The CNN pools the time dimension by 4, then by 8 (total 32).
        cnn_feature_dim = f2 * (samples // 32)

        # Classifier takes concatenated CNN + GRU + TCN features
        self.classifier = nn.Linear(cnn_feature_dim + lstm_hidden + tcn_channels, nb_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (Batch, 1, Channels, Samples)

        # 1. Extract CNN features
        cnn_out = self.features(x)  # Shape: (Batch, f2, 1, T)

        # Flatten CNN features
        cnn_features = cnn_out.view(cnn_out.size(0), -1) # Shape: (Batch, f2 * T)

        # Prepare temporal input -> remove spatial dim: (Batch, f2, T)
        temporal_input = cnn_out.squeeze(2)

        # 2. Extract GRU (LSTM) features
        lstm_input = temporal_input.permute(0, 2, 1)  # GRU needs (Batch, T, Features)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out[:, -1, :]            # Shape: (Batch, lstm_hidden)

        # 3. Extract TCN features
        # TCN natively takes (Batch, Channels, Time), which perfectly matches temporal_input
        tcn_out = self.tcn(temporal_input)            # Shape: (Batch, tcn_channels, 1)
        tcn_features = tcn_out.squeeze(2)             # Shape: (Batch, tcn_channels)

        # 4. Concatenate CNN, GRU, and TCN features
        combined_features = torch.cat((cnn_features, lstm_features, tcn_features), dim=1)

        # 5. Classify
        out = self.classifier(combined_features)
        return out
# 2. Inherit from BaseClassifier
class EEGNetClassifier(BaseClassifier):
    def __init__(self, nb_classes, chans, samples, lr=0.001, epochs=50, batch_size=32, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        # self.model = EEGNetModule(nb_classes=nb_classes, chans=chans, samples=samples).to(self.device)
        # self.model = EEGNet_CNN_LSTM_Concat(nb_classes=nb_classes, chans=chans, samples=samples).to(self.device)
        self.model = EEGNet_CNN_LSTM_TCN_Concat(nb_classes=nb_classes, chans=chans, samples=samples).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def name(self):
        return "EEG-Net"

    def train(self, x_train, y_train):
        """
        x_train: numpy array of shape (trials, channels, samples)
        y_train: numpy array of shape (trials,)
        """
        self.model.train()

        y_train_zero_indexed = y_train - 1

        # Convert to Tensors
        x_tensor = torch.from_numpy(x_train).float()
        y_tensor = torch.from_numpy(y_train_zero_indexed).long()

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Optional: print(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss/len(loader)}")

    def predict(self, x_test):
        self.model.eval()
        x_tensor = torch.from_numpy(x_test).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs, 1)

        res = predicted.cpu().numpy()

        # FIX: Shift labels back from [0, 1] to [1, 2]
        # so the output matches your original dataset labels
        return res + 1

class EEGNetClassifierMinimized(BaseClassifier):
    def __init__(self, nb_classes, chans, samples, lr=0.001, epochs=50, batch_size=32, dropout_rate=0.5, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        # self.model = EEGNetModule(nb_classes=nb_classes, chans=chans, samples=samples, dropout_rate=dropout_rate).to(self.device)
        self.model = EEGNetModuleMinimized(nb_classes=nb_classes, chans=chans, samples=samples).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    @property
    def name(self):
        return "EEG-Net"

    def train(self, x_train, y_train):
        """
        x_train: numpy array of shape (trials, channels, samples)
        y_train: numpy array of shape (trials,)
        """
        self.model.train()

        y_train_zero_indexed = y_train - 1

        # Convert to Tensors
        x_tensor = torch.from_numpy(x_train).float()
        y_tensor = torch.from_numpy(y_train_zero_indexed).long()

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Optional: print(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss/len(loader)}")

    def predict(self, x_test):
        self.model.eval()
        x_tensor = torch.from_numpy(x_test).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs, 1)

        res = predicted.cpu().numpy()

        # FIX: Shift labels back from [0, 1] to [1, 2]
        # so the output matches your original dataset labels
        return res + 1

class AutoTuningEEGNet(BaseClassifier):
    def __init__(self, chans, samples, nb_classes=2, grid=None):
        self.chans = chans
        self.samples = samples
        self.nb_classes = nb_classes
        # Default grid if none provided
        self.grid = grid if grid else {
            'lr': [0.01, 0.001],
            'dropout_rate': [0.25, 0.5],
            'batch_size': [16, 32]
        }
        self.best_model = None
        self.best_params = None

    @property
    def name(self):
        return "Auto_Tuned_EEGNet"

    def _get_combinations(self):
        """Converts dict grid into list of param dictionaries"""
        keys = self.grid.keys()
        for values in product(*self.grid.values()):
            yield dict(zip(keys, values))

    def train(self, X, y):
        best_score = -float('inf')

        # We use a simple 3-fold split to find the best params (matching your SVM cv=3)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for params in self._get_combinations():
            # For each param set, evaluate using internal validation
            fold_scores = []

            # For speed, you might want to only use 1 fold for deep learning tuning
            # but here we follow your SVM logic (cv=3)
            for train_idx, val_idx in skf.split(X, y):
                # Initialize a fresh model instance with current grid params
                model = EEGNetClassifierMinimized(
                    nb_classes=self.nb_classes,
                    chans=self.chans,
                    samples=self.samples,
                    lr=params.get('lr', 0.001),
                    batch_size=params.get('batch_size', 32),
                    dropout_rate=params.get('dropout_rate', 0.5) # Ensure EEGNetModule accepts this
                )

                # Train on fold training set
                model.train(X[train_idx], y[train_idx])

                # Evaluate on fold validation set
                preds = model.predict(X[val_idx])
                acc = np.mean(preds == y[val_idx])
                fold_scores.append(acc)
                break # Remove this break to do full 3-fold CV (will be 3x slower)

            avg_acc = np.mean(fold_scores)

            if avg_acc > best_score:
                best_score = avg_acc
                self.best_params = params
                self.best_model = model # Store the trained model

        # Final step: Retrain the best configuration on ALL data (X, y)
        # matching GridSearchCV's refit=True behavior
        self.best_model.train(X, y)

    def predict(self, X):
        if self.best_model is None:
            raise Exception("Model not trained yet.")
        return self.best_model.predict(X)