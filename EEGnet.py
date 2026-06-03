from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
import numpy as np

from interfaces import BaseClassifier
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