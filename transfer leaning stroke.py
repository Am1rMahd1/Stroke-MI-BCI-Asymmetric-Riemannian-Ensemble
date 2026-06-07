import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
import mne

from transfer_leaning_physionet import TinyEEGNet

# ==========================================
# 0. DETERMINISTIC SEEDING (Your Configuration)
# ==========================================
seed = 128
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Import your architecture from your pre-training script

# ==========================================
# 1. CORRECTED CUSTOM DATA LOAD PIPELINE
# ==========================================
def load_your_custom_dataset(subject_id="50", channels=None):
    edf_file = f"/media/amirmahdi/129699CC9699B0A9/edffile/sub-{subject_id}/eeg/sub-{subject_id}_task-motor-imagery_eeg.edf"
    csv_file = "data/task-motor-imagery_events.csv"

    # Load raw file
    raw = mne.io.read_raw_edf(edf_file, preload=True, eog=['HEOL', 'HEOR'], stim_channel=[''], verbose=False)

    # --- FIX HERE: Map your uppercase FP1/FP2 to match the pre-trained Fp1/Fp2 ---
    raw.rename_channels({
        'HEOL' : 'EOG1',
        'HEOR' : 'EOG2',
        'FP1'  : 'Fp1',
        'FP2'  : 'Fp2'
    }, verbose=False)

    raw.set_channel_types({'EOG1' : 'eog', 'EOG2' : 'eog'}, verbose=False)
    raw.drop_channels(ch_names=[''], on_missing='ignore')

    # raw.pick_channels(ch_names=channels, verbose=False)
    raw.pick_types(eeg=True, eog=False, meg=False, stim=False, exclude='bads', verbose=False)
    raw.pick_channels(ch_names=channels, verbose=False)
    # Common reference and type isolation
    raw.set_eeg_reference(ref_channels=['Cz'], verbose=False)

    # --- FIX 1: Removed restrictive channel picking ---
    # We leave all EEG channels active here so that the model can pick its 16 matching channels later.

    # Bandpass Filtering & Resampling
    raw.filter(l_freq=4.0, h_freq=30.0, method='iir', iir_params=dict(order=2, ftype='butter'), verbose=False)
    raw.resample(500, verbose=False)

    # Set spatial montage
    montage = mne.channels.make_standard_montage(kind='standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)

    # Annotation Processing via Pandas
    events_df = pd.read_csv(csv_file)
    motor_imagery_df = events_df[events_df['value'] == 2]

    # Convert ms timings to seconds for MNE
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

    # Epoch configuration
    custom_mapping = {'left_hand': 1, 'right_hand': 2}
    events, event_id = mne.events_from_annotations(raw, event_id=custom_mapping, verbose=False)

    # UPDATE: Include the 1.5 seconds of rest BEFORE the imagery cue as a baseline
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-1.5,        # Start 1.5 seconds BEFORE the motor imagery video starts (during value==1)
        tmax=4.0,         # Run until the end of the 4-second motor imagery video
        baseline=(-1.5, 0), # Use that pre-cue rest period to normalize the signal variance
        preload=True,
        proj=True,
        verbose=False,
    )

    y_labels = epochs.events[:, -1]
    mask = np.isin(y_labels, [1, 2])

    # Isolate valid trials
    epochs_valid = epochs[mask]
    y_valid = y_labels[mask]

    # --- FIX 2: Re-indexing target labels for PyTorch Loss Compatibility ---
    # Maps class labels from [1, 2] to [0, 1]
    y_valid = (y_valid - 1).astype(np.int64)

    return epochs_valid, y_valid

# ==========================================
# 2. TRANSFER LEARNING & FINE-TUNING PIPELINE
# ==========================================
# Helper function to segment trials into overlapping sliding windows
def segment_trials(X, y, window_size, step_size):
    """
    Slices trials into overlapping windows.
    Tracks 'trial_indices' to map windows back to their original trial for validation voting.
    """
    n_trials, n_chans, n_samples = X.shape
    X_sliced, y_sliced, trial_indices = [], [], []

    for i in range(n_trials):
        start = 0
        while start + window_size <= n_samples:
            end = start + window_size
            X_sliced.append(X[i, :, start:end])
            y_sliced.append(y[i])
            trial_indices.append(i) # Identifies which original trial this window belongs to
            start += step_size

    return np.array(X_sliced), np.array(y_sliced), np.array(trial_indices)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing fine-tuning validation on device: {device}")

    # 1. Load Pre-training metadata
    checkpoint = torch.load('eegnet_pretrained_base.pt')
    pretrained_channels = checkpoint['channels']

    # 2. Extract and match target clinical dataset
    user_epochs, y_user = load_your_custom_dataset(subject_id="50", channels=pretrained_channels)

    # user_epochs.pick(pretrained_channels)
    X_user = user_epochs.get_data(copy=True).astype(np.float32)

    print(f"Target dataset prepared. Unified shape footprint: {X_user.shape}")

    # ==========================================
    # DATA AUGMENTATION CONFIGURATION
    # ==========================================
    # If your pretrained model explicitly demands 4-second inputs (2000 samples),
    # keep WINDOW_SIZE = 2000 to avoid loading shape errors.
    WINDOW_SIZE = 1000  # 2 seconds @ 500 Hz
    STEP_SIZE = 250     # 0.5 seconds overlap step
    USE_NOISE_AUG = True # Alternate augmentation if WINDOW_SIZE must remain 2000

    # ==========================================
    # STRATIFIED SHUFFLE CROSS-VALIDATION SETUP
    # ==========================================
    sss = StratifiedShuffleSplit(n_splits=10, train_size=0.6, random_state=seed)

    cv_accuracies, cv_precisions, cv_sensitivities, cv_kappas = [], [], [], []

    print("\nBeginning Stratified Shuffle 10-Fold Cross-Validation with Window Augmentation...")
    print("---------------------------------------------------------------------------------")

    for fold, (train_idx, val_idx) in enumerate(sss.split(X_user, y_user)):
        print(f"\n--- Processing Fold {fold + 1}/10 ---")

        # Partition raw trials
        X_train_raw, y_train_raw = X_user[train_idx], y_user[train_idx]
        X_val_raw, y_val_raw = X_user[val_idx], y_user[val_idx]

        # Apply Sliding Window Augmentation to Training Set
        if WINDOW_SIZE < X_user.shape[2]:
            X_train, y_train, _ = segment_trials(X_train_raw, y_train_raw, WINDOW_SIZE, STEP_SIZE)
            X_val, y_val, val_trial_map = segment_trials(X_val_raw, y_val_raw, WINDOW_SIZE, STEP_SIZE)
            current_samples = WINDOW_SIZE
            print(f"Augmented training data from {len(X_train_raw)} trials into {len(X_train)} windows.")
        else:
            # Fallback if window slicing is skipped
            X_train, y_train = X_train_raw, y_train_raw
            X_val, y_val = X_val_raw, y_val_raw
            val_trial_map = np.arange(len(X_val_raw))
            current_samples = X_user.shape[2]

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) # Increased batch size slightly for augmented data

        # Re-instantiate model using the active sample length
        # Re-instantiate model using the active sample length (e.g., 1000)
        model = TinyEEGNet(nb_classes=2, Chans=X_user.shape[1], Samples=current_samples).to(device)

        # =====================================================================
        # FIX: Filter out classifier weights to allow different window sizes
        # =====================================================================
        # Extract the pre-trained weights dictionary
        pretrained_dict = checkpoint['model_state_dict'].copy()

        # Remove the classifier weights because their shape depends on the window size
        pretrained_dict.pop('classifier.weight', None)
        pretrained_dict.pop('classifier.bias', None)

        # Load the convolutional layers (strict=False ignores the missing classifier keys)
        model.load_state_dict(pretrained_dict, strict=False)
        # =====================================================================

        # Safely initialize your new, properly-sized classifier layer
        if hasattr(model.classifier, 'weight'):
            nn.init.xavier_uniform_(model.classifier.weight)
            if model.classifier.bias is not None:
                nn.init.constant_(model.classifier.bias, 0.0)

        # Apply Transfer Learning Layer Strategy
        for param in model.block1_temporal.parameters(): param.requires_grad = False
        for param in model.block1_spatial.parameters(): param.requires_grad = False
        for param in model.block2.parameters(): param.requires_grad = False
        for param in model.classifier.parameters(): param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)

        def freeze_bn_statistics(m):
            if m.__class__.__name__.find('BatchNorm') != -1:
                m.eval()

        # Training phase
        for epoch in range(60): # Reduced slightly from 90 since dataset size is now multiplied
            model.train()
            model.apply(freeze_bn_statistics)

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)

                # Optional Additive White Gaussian Noise Augmentation
                if USE_NOISE_AUG:
                    noise = torch.randn_like(bx) * 0.01
                    bx = bx + noise

                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # Validation phase using Ensemble Voting across trial windows
        model.eval()
        with torch.no_grad():
            bx_val = torch.tensor(X_val).to(device)
            out_val = model(bx_val)
            # Get soft probabilities instead of hard predictions immediately
            probs_val = torch.softmax(out_val, dim=1).cpu().numpy()

        # Reconstruct trial-level predictions from window fragments
        n_val_trials = len(X_val_raw)
        y_pred = np.zeros(n_val_trials)

        for i in range(n_val_trials):
            # Find all windows that originated from raw validation trial 'i'
            window_indices = np.where(val_trial_map == i)[0]
            # Average the probabilities across those windows (Ensemble Voting)
            mean_probabilities = np.mean(probs_val[window_indices], axis=0)
            y_pred[i] = np.argmax(mean_probabilities)

        # Compute performance metrics on true un-augmented target labels
        fold_acc = accuracy_score(y_val_raw, y_pred)
        fold_prec = precision_score(y_val_raw, y_pred, zero_division=0)
        fold_sens = recall_score(y_val_raw, y_pred, zero_division=0)
        fold_kappa = cohen_kappa_score(y_val_raw, y_pred)

        print(f"Fold {fold + 1} Metrics -> Acc: {fold_acc*100:.1f}% | Prec: {fold_prec*100:.1f}% | Sens: {fold_sens*100:.1f}% | Kappa: {fold_kappa:.3f}")

        cv_accuracies.append(fold_acc)
        cv_precisions.append(fold_prec)
        cv_sensitivities.append(fold_sens)
        cv_kappas.append(fold_kappa)

    # ==========================================
    # FINAL CROSS-VALIDATION SUMMARY REPORT
    # ==========================================
    print("\n=========================================================")
    print("      AVERAGE 10-FOLD SHUFFLED CV PERFORMANCE            ")
    print("=========================================================")
    print(f"Mean Accuracy:    {np.mean(cv_accuracies) * 100:.2f}%  (± {np.std(cv_accuracies) * 100:.2f}%)")
    print(f"Mean Precision:   {np.mean(cv_precisions) * 100:.2f}%")
    print(f"Mean Sensitivity: {np.mean(cv_sensitivities) * 100:.2f}%")
    print(f"Mean Cohen Kappa: {np.mean(cv_kappas):.4f}")
    print("=========================================================")