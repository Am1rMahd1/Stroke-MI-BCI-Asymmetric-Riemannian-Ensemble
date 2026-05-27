import os
import random

import mlflow
import numpy as np
import scipy
import torch
import scipy.io as sio
import mne

from EEGAugmentor import EEGAugmentor2
from classifiers import SVMClassifier, AutoTuningSVMClassifier
from config.config import Config
from config.model_parameters import SVM_PARAMS, SVM_GRID
from extractors import FBCSPExtractor
from splitters import IntraShuffleSplitter, IntraKFoldSplitter
from train_classifiers import calculate_metrics

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

mat_file = "/media/amirmahdi/NewVolume/matfile/BCICIV_calib_ds1b_1000Hz.mat"
data = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)

nfo = data['nfo']
cnt = data['cnt']
mrk = data['mrk']

# ۲. ایجاد اطلاعات کانال‌ها و فرکانس برای MNE
# توجه: MNE داده را به صورت (Channels, Samples) می‌خواهد، پس باید cnt را ترانهاده کنیم (.T)
ch_names = list(nfo.clab)
sfreq = nfo.fs  # ۱۰۰۰ هرتز
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# ایجاد شیء Raw
raw = mne.io.RawArray(cnt.T, info)

# ۳. تنظیم مونتاژ (مختصات الکترودها)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='ignore')

# ۴. پیش‌پردازش (فیلتر و رفرنس‌دهی)
raw.set_eeg_reference(ref_channels=['Cz'], verbose=False)

# انتخاب کانال‌های مورد نظر شما (اگر در این لیست باشند)
target_channels = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']
# اطمینان از اینکه کانال‌ها در دیتاست وجود دارند
available_channels = [ch for ch in target_channels if ch in raw.ch_names]
raw.pick_channels(available_channels)

# فیلتر کردن (۴ تا ۳۰ هرتز)
raw.filter(
    l_freq=4.0,
    h_freq=30.0,
    method='fir',
    # iir_params=dict(order=2, ftype='butter'),
    verbose=False
)

# تغییر فرکانس نمونه‌برداری به ۵۰۰ هرتز (طبق کد قبلی شما)
raw.resample(600, verbose=False)

# ۵. مدیریت ایونت‌ها (Annotations)
# در این دیتاست mrk.pos بر حسب سمپل است. برای آنوتیشن باید به ثانیه تبدیل شود.
onsets = mrk.pos / sfreq  # تبدیل ایندکس به ثانیه
durations = [4.0] * len(onsets)  # معمولاً هر تریال ۴ ثانیه است

# نگاشت لیبل‌ها: معمولاً ۱ برای چپ و ۱- یا ۲ برای راست است
# بر اساس nfo.classes که ['left', 'right'] بود:
description = []
for label in mrk.y:
    if label == 1:
        description.append('left_hand')
    else:
        description.append('right_hand')

annotations = mne.Annotations(onset=onsets, duration=durations, description=description)
raw.set_annotations(annotations)

# ۶. ایجاد Epochs
custom_mapping = {'left_hand': 1, 'right_hand': 2}
events, event_id = mne.events_from_annotations(raw, event_id=custom_mapping, verbose=False)

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=0.0,   # شروع از نیم ثانیه بعد از محرک (برای حذف اثر لحظه‌ای ویژوال)
    tmax=4.0,   # تا ثانیه ۴
    baseline=None,
    preload=True,
    proj=True,
    verbose=False,
)

y_labels = epochs.events[:, -1]
mask = np.isin(y_labels, [1, 2])
x_data = epochs.get_data(copy=False)[mask]
y_labels = y_labels[mask]

_, channels, samples = x_data.shape

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Mat file")

with mlflow.start_run(run_name=f"mat file"):
    subject_avg_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }
    fold_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }
    splitter = IntraKFoldSplitter(n_splits=10, random_state=seed)
    # splitter = IntraShuffleSplitter(n_splits=10, train_size=0.6, random_state=seed)
    for train_idx, test_idx in splitter.get_splits(x_data, y_labels):
        # clf = EEGNetClassifier(nb_classes=2,chans=channels, samples=samples, epochs=100)
        clf = SVMClassifier(SVM_PARAMS)

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
        X_train_raw, y_train = augmentor.augment(X_train_raw, y_train)
        # Now X_train_raw and y_train are doubled in size
        # ==========================================

        config =Config()
        fold_extractor = FBCSPExtractor(600, config.FREQ_BANDS, config.N_COMPONENTS, 'ledoit_wolf')
        fold_extractor.fit(X_train_raw, y_train)

        X_train_features = fold_extractor.transform(X_train_raw)
        X_test_features = fold_extractor.transform(X_test_raw)
        # X_train_features = X_train_raw
        # X_test_features = X_test_raw

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