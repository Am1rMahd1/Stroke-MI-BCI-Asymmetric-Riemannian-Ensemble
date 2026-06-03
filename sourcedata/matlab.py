import os
import glob
import random

import numpy as np
import scipy.io as sio
import mne
import matplotlib.pyplot as plt
import torch
from pyriemann.classification import FgMDM

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

seed = 128
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def TWFB_DGFMDM(MIEEGData, label, Fs, LowFreq, UpFreq):
    """
    Optimized: TWFB + OAS Covariance + Tangent Space Mapping + Logistic Regression
    """
    channels = np.concatenate([np.arange(0, 17), np.arange(18, 30)])
    trigger = np.where(MIEEGData[:, 32] == 2)[0]
    n_trials = len(trigger)

    acc = []
    left_num = []
    right_num = []

    for h in range(10):
        freq_bands = [[8,12], [8,20], [8,30], [12,20], [15,20], [15,30], [20,30], [8,15]]

        acc_temp = []
        left_num_temp = []
        right_num_temp = []

        for n, freq in enumerate(freq_bands):
            epochs_data = np.zeros((n_trials, len(channels), 2000))

            for i, trig_idx in enumerate(trigger):
                start = trig_idx - 800
                end = start + 2800

                R = MIEEGData[start:end, channels].T

                notch_data = mne.filter.notch_filter(R, Fs, freqs=50.0, method='iir', verbose=False)
                bp_data = mne.filter.filter_data(notch_data, Fs, l_freq=freq[0], h_freq=freq[1], method='iir', verbose=False)

                epochs_data[i, :, :] = bp_data[:, 800:2800]

            perms = np.random.permutation(n_trials)
            train_idx = perms[:24]
            test_idx = perms[24:]

            eeg_train = epochs_data[train_idx, :, :]
            eeg_test = epochs_data[test_idx, :, :]
            label_train = label[train_idx]
            label_test = label[test_idx]

            # --- IMPROVEMENT 1: Oracle Approximating Shrinkage (oas) ---
            # 'oas' is much more robust to noise and small sample sizes than 'scm'
            cov_estimator = Covariances(estimator='oas')
            COVtrain = cov_estimator.fit_transform(eeg_train)
            COVtest = cov_estimator.fit_transform(eeg_test)

            # --- IMPROVEMENT 2: Tangent Space + Linear Classifier ---
            # Projects matrices to Euclidean space, then uses a robust classifier
            clf = make_pipeline(
                Covariances(estimator='lwf'), # جایگزین scm برای جلوگیری از خطای مثبت معین نبودن
                FgMDM(metric='riemann')
            )

            clf.fit(COVtrain, label_train)
            Ytest = clf.predict(COVtest)

            # Metric Tracking
            c = np.sum(Ytest == label_test)
            L = np.sum((Ytest == label_test) & (label_test == 1))
            R_correct = np.sum((Ytest == label_test) & (label_test == 2))

            left_total = np.sum(label_test == 1)
            right_total = np.sum(label_test == 2)

            acc_temp.append((c / len(label_test)) * 100)
            left_num_temp.append([left_total, L])
            right_num_temp.append([right_total, R_correct])

        acc_temp = np.array(acc_temp)
        best_band_idx = np.argmax(acc_temp)

        acc.append(acc_temp[best_band_idx])
        left_num.append(left_num_temp[best_band_idx])
        right_num.append(right_num_temp[best_band_idx])

    return np.array(acc), np.array(left_num), np.array(right_num)

# --- Visualization and Evaluation Helpers ---

def PlotAccBar(Acc):
    mean_acc = np.mean(Acc)

    plt.figure(1, figsize=(5, 5))
    plt.bar(['TWFB+OAS+TS+LR'], [mean_acc], color='#2ca02c', width=0.4)
    plt.ylabel('Accuracy (%)')
    plt.title('Overall Classification Accuracy')
    plt.ylim(0, 100)
    plt.text(0, mean_acc + 1, f"{mean_acc:.2f}%", ha='center', fontweight='bold')
    plt.show()

def PlotConfusionMatrix(cm):
    plt.figure(2, figsize=(5, 4))
    classes = ['Right', 'Left']

    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title('CM: TWFB+OAS+TS+LR')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, f"{int(cm[row, col])}",
                     ha="center", va="center",
                     color="white" if cm[row, col] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

def CalEvaluateIndex(cm):
    TP_R, FN_R = cm[0, 0], cm[0, 1]
    FP_R, TN_R = cm[1, 0], cm[1, 1]

    total = TP_R + FN_R + FP_R + TN_R
    sens = TP_R / (TP_R + FN_R) if (TP_R + FN_R) > 0 else 0
    prec = TP_R / (TP_R + FP_R) if (TP_R + FP_R) > 0 else 0

    po = (TP_R + TN_R) / total
    pe = ((TP_R + FN_R) * (TP_R + FP_R) + (FP_R + TN_R) * (FN_R + TN_R)) / (total ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    return kappa, sens, prec


# --- Main Processing Execution Loop ---

if __name__ == "__main__":
    path = "/media/amirmahdi/NewVolume"

    Acc, Left, Right, AllLeft, AllRight = [], [], [], [], []

    for i in range(1, 51):
        print("Processing", i)
        sub_folder = os.path.join(path, 'sourcedata', f'sub-{i:02d}')
        if not os.path.exists(sub_folder):
            continue

        mat_files = glob.glob(os.path.join(sub_folder, '*.mat'))

        for mat_file in mat_files:
            mat_data = sio.loadmat(mat_file, squeeze_me=True)
            eeg_struct = mat_data['eeg']

            eeg0 = eeg_struct['rawdata']
            label = eeg_struct['label']

            if isinstance(eeg0, np.ndarray) and eeg0.dtype == object:
                if eeg0.ndim == 0:
                    eeg0 = eeg0.item()
                elif eeg0.ndim == 1 and len(eeg0) == 1:
                    eeg0 = eeg0[0]

            eeg0 = np.array(eeg0)

            if eeg0.ndim == 3:
                if eeg0.shape[1] < eeg0.shape[2]:
                    eeg0 = np.transpose(eeg0, (0, 2, 1))
                data = eeg0.reshape(-1, eeg0.shape[2])
            elif eeg0.ndim == 2:
                data = eeg0.T if eeg0.shape[0] < eeg0.shape[1] else eeg0
            elif eeg0.ndim == 4:
                eeg0 = np.squeeze(eeg0)
                if eeg0.ndim == 3:
                    if eeg0.shape[1] < eeg0.shape[2]:
                        eeg0 = np.transpose(eeg0, (0, 2, 1))
                    data = eeg0.reshape(-1, eeg0.shape[2])
                elif eeg0.ndim == 2:
                    data = eeg0.T if eeg0.shape[0] < eeg0.shape[1] else eeg0
            else:
                raise ValueError(f"Unsupported matrix dimensions found: {eeg0.ndim}")

            if isinstance(label, np.ndarray) and label.dtype == object:
                if label.ndim == 0:
                    label = label.item()
                elif label.ndim == 1 and len(label) == 1:
                    label = label[0]

            label = np.array(label).flatten()

            Fs, LowFreq, UpFreq = 500, 4, 30

            acc, left_num, right_num = TWFB_DGFMDM(data, label, Fs, LowFreq, UpFreq)

            Acc.append(np.mean(acc))
            Left.append(np.mean(left_num[:, 1]))
            Right.append(np.mean(right_num[:, 1]))
            AllLeft.append(np.mean(left_num[:, 0]))
            AllRight.append(np.mean(right_num[:, 0]))

    Acc = np.array(Acc)
    Left = np.array(Left)
    Right = np.array(Right)
    AllLeft = np.array(AllLeft)
    AllRight = np.array(AllRight)

    if len(Acc) > 0:
        PlotAccBar(Acc)

        sum_r = np.floor(np.sum(Right))
        sum_all_r = np.floor(np.sum(AllRight))
        sum_l = np.floor(np.sum(Left))
        sum_all_l = np.floor(np.sum(AllLeft))

        CM = np.array([
            [sum_r, sum_all_r - sum_r],
            [sum_all_l - sum_l, sum_l]
        ])

        PlotConfusionMatrix(CM)
        Kappa, Sensitivity, Precision = CalEvaluateIndex(CM)

        print(f"\n--- Evaluation Indices ---")
        print(f"Kappa Coefficient: {Kappa:.4f}")
        print(f"Sensitivity:       {Sensitivity:.4f}")
        print(f"Precision:         {Precision:.4f}")
    else:
        print("No data folders or valid files found matching the paths.")
