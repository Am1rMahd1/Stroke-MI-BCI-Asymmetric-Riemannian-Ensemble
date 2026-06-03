import os
import random

import mlflow
import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from Nature import EEGTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import copy
from splitters import IntraKFoldSplitter, IntraShuffleSplitter


class EEGAugmentor2:
    """
    Handles data augmentation for EEG signals.
    Must be applied ONLY to training data after splitting.
    """
    def __init__(self, method='mixup', max_shift=10, multiplier=3):
        # method: 'mixup' or 'time_shift'
        # max_shift: maximum samples to shift (only used for 'time_shift')
        # multiplier: Total size of dataset relative to original (e.g., 3 means 1 original + 2 augmented sets)
        self.method = method
        self.max_shift = max_shift
        self.multiplier = multiplier

    def _intra_class_mixup(self, X, y):
        """Averages pairs of epochs from the same class ($X_{new} = 0.5 X_1 + 0.5 X_2$)."""
        X_mixed = np.zeros_like(X)
        classes = np.unique(y)

        for c in classes:
            idx = np.where(y == c)[0]
            for i in idx:
                j = np.random.choice(idx)
                X_mixed[i] = 0.5 * X[i] + 0.5 * X[j]

        return X_mixed

    def _time_shift(self, X):
        """Randomly shifts the EEG signal forward or backward in time."""
        X_shifted = np.zeros_like(X)

        for i in range(len(X)):
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            X_shifted[i] = np.roll(X[i], shift=shift, axis=1)

        return X_shifted

    def augment(self, X_train, y_train, groups_train=None):
        """
        Generates augmented data and appends it to the original training data.
        Returns the expanded dataset.
        """
        X_list = [X_train]
        y_list = [y_train]
        if groups_train is not None:
            groups_list = [groups_train]

        # Generate (multiplier - 1) new sets of data
        for _ in range(self.multiplier - 1):
            if self.method == 'mixup':
                X_new = self._intra_class_mixup(X_train, y_train)
            elif self.method == 'time_shift':
                X_new = self._time_shift(X_train)
            else:
                raise ValueError("Method must be 'mixup' or 'time_shift'")

            X_list.append(X_new)
            y_list.append(y_train)
            if groups_train is not None:
                groups_list.append(groups_train)

        # Concatenate everything together
        X_augmented = np.vstack(X_list)
        y_augmented = np.concatenate(y_list)

        if groups_train is not None:
            groups_augmented = np.concatenate(groups_list)
            return X_augmented, y_augmented, groups_augmented

        return X_augmented, y_augmented

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
class EEGNetClassifier(): # (اگر BaseClassifier دارید، آن را داخل پرانتز قرار دهید)
    def __init__(self, nb_classes, chans, samples, lr=0.001, epochs=1000, batch_size=72,
                 validate_ratio=0.3, number_aug=3, number_seg=8, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.chans = chans
        self.samples = samples

        # Hyperparameters specific to ExP logic
        self.validate_ratio = validate_ratio
        self.number_aug = number_aug
        self.number_seg = number_seg
        self.lr = lr
        self.b1 = 0.5
        self.b2 = 0.999

        # Instantiate Model
        self.model = EEGTransformer(num_classes=nb_classes, num_channels=chans).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # To track the best model
        self.best_model_wts = None
        self.best_epoch = 0

    @property
    def name(self):
        return "EEG-Net-Transformer"

    def interaug(self, timg, label):
        """ Segmentation and Reconstruction (S&R) Data Augmentation """
        aug_data = []
        aug_label = []
        # Calculate how many records to generate per class
        number_records_by_augmentation = self.number_aug * max(1, int(self.batch_size / self.nb_classes))
        number_segmentation_points = self.samples // self.number_seg

        for clsAug in range(self.nb_classes):
            cls_idx = np.where(label == clsAug)[0] # zero-indexed label expected here

            # If a class is missing in the current split, skip it
            if len(cls_idx) == 0:
                continue

            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.chans, self.samples))
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    start_pt = rj * number_segmentation_points
                    end_pt = (rj + 1) * number_segmentation_points

                    tmp_aug_data[ri, :, :, start_pt:end_pt] = tmp_data[rand_idx[rj], :, :, start_pt:end_pt]

            aug_data.append(tmp_aug_data)
            # ایجاد یک آرایه از لیبل‌های کلاس فعلی به تعداد دقیق دیتای مصنوعی تولید شده
            aug_label.append(np.full(number_records_by_augmentation, clsAug))

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)

        # Shuffle Augmented Data
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        # Convert to Tensors
        aug_data = torch.from_numpy(aug_data).float().to(self.device)
        aug_label = torch.from_numpy(aug_label).long().to(self.device)

        return aug_data, aug_label

    def train(self, x_train, y_train):
        # 1. Protect signal shape: Ensure 4D (Trials, 1, Channels, Samples)
        if len(x_train.shape) == 3:
            x_train_4d = np.expand_dims(x_train, axis=1)
        else:
            x_train_4d = x_train

        y_train_zero_indexed = y_train - 1

        # 2. Setup Dataset
        x_tensor = torch.from_numpy(x_train_4d).float()
        y_tensor = torch.from_numpy(y_train_zero_indexed).long()
        dataset = TensorDataset(x_tensor, y_tensor)

        # 3. Setup Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        min_val_loss = float('inf')

        # 4. Training Loop
        for epoch in range(self.epochs):
            self.model.train()
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            val_data_list = []
            val_label_list = []

            for inputs, labels in loader:
                num_sample = inputs.shape[0]
                num_validate = int(self.validate_ratio * num_sample)

                # Split batch exactly like ExP logic
                if num_validate > 0:
                    train_data = inputs[:-num_validate]
                    train_label = labels[:-num_validate]
                    val_data_list.append(inputs[-num_validate:])
                    val_label_list.append(labels[-num_validate:])
                else:
                    train_data = inputs
                    train_label = labels

                train_data = train_data.to(self.device)
                train_label = train_label.to(self.device)

                # Apply Augmentation
                aug_data, aug_label = self.interaug(x_train_4d, y_train_zero_indexed)

                # Concat real training data with artificial data
                batch_data = torch.cat((train_data, aug_data))
                batch_label = torch.cat((train_label, aug_label))

                # Forward & Backward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)

                # --- هوشمندسازی: پیدا کردن خودکار تنسور کلاس‌بندی ---
                if isinstance(outputs, tuple):
                    for out in outputs:
                        if isinstance(out, torch.Tensor) and out.dim() == 2:
                            outputs = out
                            break
                # ---------------------------------------------------

                loss = self.criterion(outputs, batch_label)
                loss.backward()
                self.optimizer.step()

            # 5. Validation Phase (End of Epoch)
            if len(val_data_list) > 0:
                self.model.eval()
                val_data = torch.cat(val_data_list).to(self.device)
                val_label = torch.cat(val_label_list).to(self.device)

                with torch.no_grad():
                    val_outputs = self.model(val_data)

                    # --- هوشمندسازی: پیدا کردن خودکار تنسور کلاس‌بندی ---
                    if isinstance(val_outputs, tuple):
                        for out in val_outputs:
                            if isinstance(out, torch.Tensor) and out.dim() == 2:
                                val_outputs = out
                                break
                    # ---------------------------------------------------

                    val_loss = self.criterion(val_outputs, val_label)

                # Checkpoint best model
                if val_loss.item() < min_val_loss:
                    min_val_loss = val_loss.item()
                    self.best_epoch = epoch
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

        # Load best model weights before ending train function
        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)

    def predict(self, x_test):
        # Ensure best weights are loaded
        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)

        self.model.eval()

        # Ensure 4D shape for test data
        if len(x_test.shape) == 3:
            x_test_4d = np.expand_dims(x_test, axis=1)
        else:
            x_test_4d = x_test

        x_tensor = torch.from_numpy(x_test_4d).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(x_tensor)

            # --- هوشمندسازی: پیدا کردن خودکار تنسور کلاس‌بندی ---
            if isinstance(outputs, tuple):
                for out in outputs:
                    if isinstance(out, torch.Tensor) and out.dim() == 2:
                        outputs = out
                        break
            # ---------------------------------------------------

            _, predicted = torch.max(outputs, 1)

        res = predicted.cpu().numpy()

        # Shift labels back from [0, 1] to [1, 2]
        return res + 1
def calculate_metrics(y_true, y_pred):
    # 1. Calculate Custom BCI Kappa (pe hardcoded to 0.5)
    p0 = accuracy_score(y_true, y_pred)
    pe = 0.5
    custom_kappa = (p0 - pe) / (1.0 - pe)

    # 2. Generate the Confusion Matrix
    # We explicitly define labels=[1, 2] so Class 1 is always at index 0
    # and Class 2 is always at index 1, preventing any accidental axis flipping.
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])

    # In scikit-learn: Rows = True Labels, Columns = Predicted Labels
    TP = cm[0, 0]  # True Class 1, Predicted Class 1 (MATLAB's confusion_matrix(1,1))
    FN = cm[0, 1]  # True Class 1, Predicted Class 2 (MATLAB's confusion_matrix(2,1))
    FP = cm[1, 0]  # True Class 2, Predicted Class 1 (MATLAB's confusion_matrix(1,2))

    # 3. Calculate Class 1 Metrics (with zero-division protection)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    return {
        "accuracy": p0,
        "precision": precision,
        "sensitivity": sensitivity,
        "kappa": custom_kappa
    }

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

subject_id = "50"
edf_file = f"/media/amirmahdi/NewVolume/edffile/sub-{subject_id}/eeg/sub-{subject_id}_task-motor-imagery_eeg.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True, eog=['HEOL', 'HEOR'], stim_channel=[''], verbose=False)
raw.rename_channels({'HEOL' : 'EOG1', 'HEOR' : 'EOG2'}, verbose=False)
raw.set_channel_types({'EOG1' : 'eog', 'EOG2' : 'eog'}, verbose=False)
raw.drop_channels(
    ch_names=[''],
    on_missing='ignore'
)
raw.set_eeg_reference(
    ref_channels='average',
    verbose=False
)
raw.pick_types(
    eeg=True,
    eog=False,
    meg=False,
    stim=False,
    exclude='bads',
    verbose=False
)
raw.pick_channels(
    # ch_names=['C3', 'C4', 'Cz', 'CP3', 'CP4', 'CPz'],
    # ch_names=['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4'],
    ch_names=['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4'],
    verbose=False
)
raw.filter(
    l_freq=4.0,
    h_freq=30.0,
    method='iir',
    # iir_params=dict(order=2, ftype='butter'),
    verbose=False,
)
raw.resample(500, verbose=False)
montage = mne.channels.make_standard_montage(
    kind='standard_1020'
)
raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
events_df = pd.read_csv("data/task-motor-imagery_events.csv")
motor_imagery_df = events_df[events_df['value'] == 2]

onset = motor_imagery_df['onset'] / 1000.0
duration = motor_imagery_df['duration'] / 1000.0

description = []
for _, row in motor_imagery_df.iterrows():
    trial_type = row['trial_type']
    if trial_type == 1:
        description.append('left_hand')
    else:
        description.append('right_hand')

annotations = mne.Annotations(onset=onset, duration=duration, description=description)
raw.set_annotations(annotations, verbose=False)

custom_mapping = {'left_hand': 1, 'right_hand': 2}
events, event_id = mne.events_from_annotations(raw, event_id=custom_mapping, verbose=False)
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=0.0,
    tmax=4.0,
    baseline=None,
    preload=True,
    proj=True,
    verbose=False,
)

# epochs.resample(500, verbose=False)
y_labels = epochs.events[:, -1]
mask = np.isin(y_labels, [1, 2])
x_data = epochs.get_data(copy=False)[mask]
y_labels = y_labels[mask]

_, channels, samples = x_data.shape

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Single Subject")

with mlflow.start_run(run_name=f"subject {subject_id}"):
    subject_avg_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }
    fold_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }
    # splitter = IntraKFoldSplitter(n_splits=10, random_state=seed)
    splitter = IntraShuffleSplitter(n_splits=10, train_size=0.6, random_state=seed)
    for train_idx, test_idx in splitter.get_splits(x_data, y_labels):
        clf = EEGNetClassifier(nb_classes=2,chans=channels, samples=samples)

        X_train_raw = x_data[train_idx]
        X_test_raw = x_data[test_idx]
        y_train = y_labels[train_idx]
        y_test = y_labels[test_idx]

        # ==========================================
        # 1.5. ADD Z-NORMALIZATION HERE
        # ==========================================
        # Calculate mean/std ONLY on the augmented training data
        # Assuming X shape is (Samples, Channels, Time)
        # We calculate mean/std per channel
        train_mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
        train_std = X_train_raw.std(axis=(0, 2), keepdims=True)

        # # Add a small epsilon to avoid division by zero
        train_std[train_std == 0] = 1e-8

        # Normalize training and testing sets
        X_train_raw = (X_train_raw - train_mean) / train_std
        X_test_raw = (X_test_raw - train_mean) / train_std

        # ==========================================
        # 1. ADD AUGMENTATION HERE (ONLY ON TRAIN)
        # ==========================================
        augmentor = EEGAugmentor2(method='time_shift', multiplier=2)
        # X_train_raw, y_train = augmentor.augment(X_train_raw, y_train)
        # Now X_train_raw and y_train are doubled in size
        # ==========================================

        # config =Config()
        # fold_extractor = FBCSPExtractor(500, config.FREQ_BANDS, config.N_COMPONENTS, None)
        # fold_extractor.fit(X_train_raw, y_train)

        # X_train_features = fold_extractor.transform(X_train_raw)
        # X_test_features = fold_extractor.transform(X_test_raw)
        X_train_features = X_train_raw[:, np.newaxis, :, :]
        X_test_features = X_test_raw[:, np.newaxis, :, :]

        clf.train(X_train_features, y_train)
        y_pred = clf.predict(X_test_features)

        metrics = calculate_metrics(y_test, y_pred)

        for k, v in metrics.items(): fold_metrics[k].append(v)

    # Average folds for THIS subject
    for k in subject_avg_metrics.keys():
        sub_mean = np.mean(fold_metrics[k])
        subject_avg_metrics[k].append(sub_mean)

    for k, v in subject_avg_metrics.items():
        mlflow.log_metric(f"mean_{k}", np.mean(v))
        mlflow.log_metric(f"std_{k}", np.std(v))