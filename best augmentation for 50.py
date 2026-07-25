import os
import random
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.feature_selection import SelectKBest, mutual_info_classif

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==========================================================
# 1. تنظیم جهت‌دهی تصادفی (Seed Management)
# ==========================================================
def set_seeds(seed: int = 128) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# ==========================================================
# 2. کلاس مدیریت انتخاب کانال مبتنی بر آناتومی
# ==========================================================
class ClinicalChannelSelector:
    BASE_CHANNELS = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']

    @classmethod
    def get_smart_channels(cls, location_str: Any, paralysis_side: Any) -> List[str]:
        if pd.isna(location_str) or not isinstance(location_str, str):
            return cls.BASE_CHANNELS.copy()

        loc = location_str.lower()
        p_side = str(paralysis_side).lower() if not pd.isna(paralysis_side) else ""

        if 'pons' in loc or 'medulla' in loc or 'cerebell' in loc:
            return ['FCz', 'C3', 'Cz', 'C4', 'CPz']

        smart_channels = cls.BASE_CHANNELS.copy()

        if 'left' in loc or 'right' in p_side:
            if any(k in loc for k in ['frontal', 'cortex', 'radiata', 'capsule', 'parietal', 'semiovale']):
                smart_channels += ['F3', 'Fz', 'P3', 'Pz']

        if 'right' in loc or 'left' in p_side:
            if any(k in loc for k in ['frontal', 'cortex', 'radiata', 'capsule', 'parietal', 'occipital', 'semiovale']):
                smart_channels += ['F4', 'Fz', 'P4', 'Pz']

        return sorted(list(set(smart_channels)))

# ==========================================================
# 3. ساختار پایه برای داده‌افزایی (Augmentation)
# ==========================================================
class BaseAugmenter(ABC):
    def __init__(self, aug_factor: int = 2, seed: int = 128):
        self.aug_factor = aug_factor
        self.seed = seed

    @abstractmethod
    def augment(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

class AdaptiveClinicalAugmenter(BaseAugmenter):
    def __init__(self, method_str: str = "Identity", aug_factor: int = 2, seed: int = 128):
        super().__init__(aug_factor=aug_factor, seed=seed)
        self.method_str = method_str

    def augment(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(self.seed)

        if self.method_str == 'Identity' or not self.method_str:
            return X_train, y_train

        X_aug, y_aug = [], []
        classes = np.unique(y_train)
        _, n_bands, n_channels, n_times = X_train.shape

        if 'AVG' in self.method_str:
            for c in classes:
                idx = np.where(y_train == c)[0]
                for _ in range(len(idx) * self.aug_factor):
                    i1, i2 = np.random.choice(idx, 2, replace=True)
                    X_aug.append((X_train[i1] + X_train[i2]) / 2.0)
                    y_aug.append(c)

        if 'NS' in self.method_str:
            std_dev = np.std(X_train, axis=(0, -1), keepdims=True)[0]
            for c in classes:
                idx = np.where(y_train == c)[0]
                for _ in range(self.aug_factor):
                    for i in idx:
                        noise = np.random.normal(0, 0.1 * std_dev, size=X_train[i].shape)
                        X_aug.append(X_train[i] + noise)
                        y_aug.append(c)

        if len(X_aug) > 0:
            X_train = np.concatenate([X_train, np.array(X_aug)], axis=0)
            y_train = np.concatenate([y_train, np.array(y_aug)], axis=0)

        return X_train, y_train

# ==========================================================
# 4. ماژول‌های استخراج ویژگی (Feature Extractors)
# ==========================================================
class BaseFeatureExtractor(BaseEstimator, TransformerMixin, ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

class StandardFBCSP(BaseFeatureExtractor):
    def __init__(self, n_bands: int, n_components: int = 4, k_features: int = 16):
        self.n_bands = n_bands
        self.n_components = n_components
        self.k_features = k_features
        self.csps_ = []
        self.selector_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.csps_ = []
        all_train_features = []

        for b in range(self.n_bands):
            X_band = X[:, b, :, :]
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True, norm_trace=False)
            csp.fit(X_band, y)
            self.csps_.append(csp)
            feat_csp = csp.transform(X_band)
            all_train_features.append(feat_csp)

        X_all_feat = np.hstack(all_train_features)
        actual_k = min(self.k_features, X_all_feat.shape[1])

        mi_func = partial(mutual_info_classif, random_state=128)
        self.selector_ = SelectKBest(score_func=mi_func, k=actual_k)
        self.selector_.fit(X_all_feat, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        all_features = []
        for b in range(self.n_bands):
            X_band = X[:, b, :, :]
            feat_csp = self.csps_[b].transform(X_band)
            all_features.append(feat_csp)

        X_all_feat = np.hstack(all_features)
        return self.selector_.transform(X_all_feat)

class ImprovedFBCSP_ASP(BaseFeatureExtractor):
    def __init__(self, n_bands: int, n_components: int = 4, n_asp_components: int = 2,
                 n_particles: int = 30, max_iter: int = 20, k_features: int = 16):
        self.n_bands = n_bands
        self.n_components = n_components
        self.n_asp_components = n_asp_components
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.k_features = k_features
        self.csps_ = []
        self.asp_filters_ = []
        self.selector_ = None

    def _optimize_asp_filter(self, covs: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        _, n_channels, _ = covs.shape
        classes = np.unique(y)

        if len(classes) != 2:
            return rng.standard_normal((n_channels,))

        idx1 = (y == classes[0])
        idx2 = (y == classes[1])

        X_p = rng.standard_normal((self.n_particles, n_channels))
        X_p /= (np.linalg.norm(X_p, axis=1, keepdims=True) + 1e-8)

        V_p = rng.standard_normal((self.n_particles, n_channels)) * 0.1

        pbest = np.copy(X_p)
        pbest_fit = np.full(self.n_particles, -1e9)

        gbest = np.copy(X_p[0])
        gbest_fit = -1e9

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                w = X_p[i]
                tmp = covs @ w
                E = np.sum(tmp * w, axis=1)
                E = np.log(np.maximum(E, 1e-8))

                m1, m2 = np.mean(E[idx1]), np.mean(E[idx2])
                v1 = np.var(E[idx1]) + 1e-6
                v2 = np.var(E[idx2]) + 1e-6

                fit_val = ((m1 - m2) ** 2) / (v1 + v2)

                if fit_val > pbest_fit[i]:
                    pbest_fit[i] = fit_val
                    pbest[i] = np.copy(w)

                if fit_val > gbest_fit:
                    gbest_fit = fit_val
                    gbest = np.copy(w)

            w_inertia = 0.729
            c1, c2 = 1.49445, 1.49445
            r1 = rng.uniform(0, 1, (self.n_particles, n_channels))
            r2 = rng.uniform(0, 1, (self.n_particles, n_channels))

            V_p = w_inertia * V_p + c1 * r1 * (pbest - X_p) + c2 * r2 * (gbest - X_p)
            X_p = X_p + V_p
            X_p /= (np.linalg.norm(X_p, axis=1, keepdims=True) + 1e-8)

        return gbest

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.csps_ = []
        self.asp_filters_ = []
        all_train_features = []

        for b in range(self.n_bands):
            X_band = X[:, b, :, :]

            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True, norm_trace=False)
            csp.fit(X_band, y)
            self.csps_.append(csp)
            feat_csp = csp.transform(X_band)

            covs = np.matmul(X_band, X_band.transpose(0, 2, 1)) / X_band.shape[-1]
            band_filters = []
            feat_asp_list = []

            for c_idx in range(self.n_asp_components):
                w_best = self._optimize_asp_filter(covs, y, seed=128 + b * 10 + c_idx)
                band_filters.append(w_best)

                tmp = covs @ w_best
                E = np.sum(tmp * w_best, axis=1)
                feat_asp_w = np.log(np.maximum(E, 1e-8)).reshape(-1, 1)
                feat_asp_list.append(feat_asp_w)

            self.asp_filters_.append(band_filters)

            feat_band = np.hstack([feat_csp] + feat_asp_list)
            all_train_features.append(feat_band)

        X_all_feat = np.hstack(all_train_features)
        actual_k = min(self.k_features, X_all_feat.shape[1])

        mi_func = partial(mutual_info_classif, random_state=128)
        self.selector_ = SelectKBest(score_func=mi_func, k=actual_k)
        self.selector_.fit(X_all_feat, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        all_features = []
        for b in range(self.n_bands):
            X_band = X[:, b, :, :]

            feat_csp = self.csps_[b].transform(X_band)

            covs = np.matmul(X_band, X_band.transpose(0, 2, 1)) / X_band.shape[-1]
            feat_asp_list = []

            for w_best in self.asp_filters_[b]:
                tmp = covs @ w_best
                E = np.sum(tmp * w_best, axis=1)
                feat_asp_w = np.log(np.maximum(E, 1e-8)).reshape(-1, 1)
                feat_asp_list.append(feat_asp_w)

            feat_band = np.hstack([feat_csp] + feat_asp_list)
            all_features.append(feat_band)

        X_all_feat = np.hstack(all_features)
        return self.selector_.transform(X_all_feat)

class StandardFBRiemann(BaseFeatureExtractor):
    def __init__(self, n_bands: int):
        self.n_bands = n_bands
        self.covs_ = []
        self.ts_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.covs_ = []
        self.ts_ = []
        for b in range(self.n_bands):
            cov = Covariances(estimator='lwf')
            ts = TangentSpace(metric='riemann')
            cov_matrices = cov.fit_transform(X[:, b, :, :])
            ts.fit(cov_matrices, y)
            self.covs_.append(cov)
            self.ts_.append(ts)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_feat = []
        for b in range(self.n_bands):
            cov_matrices = self.covs_[b].transform(X[:, b, :, :])
            feat_b = self.ts_[b].transform(cov_matrices)
            X_feat.append(feat_b)
        return np.hstack(X_feat)

# ==========================================================
# 5. ساختار انتزاعی و پیاده‌سازی حالت‌های Ablation Study (OCP)
# ==========================================================
class BaseAblationStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def use_smart_channels(self) -> bool:
        pass

    @property
    @abstractmethod
    def use_augmentation(self) -> bool:
        pass

    @abstractmethod
    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        pass

class Config1_Baseline(BaseAblationStrategy):
    @property
    def name(self) -> str: return "1_FBCSP (Base Channels, No Aug)"
    @property
    def use_smart_channels(self) -> bool: return False
    @property
    def use_augmentation(self) -> bool: return False

    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        n_comp = min(4, X_tr.shape[2])
        pipe = Pipeline([
            ('fbcsp', StandardFBCSP(n_bands=n_bands, n_components=n_comp, k_features=16)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        pipe.fit(X_tr, y_tr)
        return pipe.predict(X_te)

class Config2_AddChannelSelection(BaseAblationStrategy):
    @property
    def name(self) -> str: return "2_FBCSP (Smart Channels, No Aug)"
    @property
    def use_smart_channels(self) -> bool: return True
    @property
    def use_augmentation(self) -> bool: return False

    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        n_comp = min(4, X_tr.shape[2])
        pipe = Pipeline([
            ('fbcsp', StandardFBCSP(n_bands=n_bands, n_components=n_comp, k_features=16)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        pipe.fit(X_tr, y_tr)
        return pipe.predict(X_te)

class Config3_AddASP(BaseAblationStrategy):
    @property
    def name(self) -> str: return "3_FBCSP-ASP (Smart Channels, No Aug)"
    @property
    def use_smart_channels(self) -> bool: return True
    @property
    def use_augmentation(self) -> bool: return False

    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        n_comp = min(4, X_tr.shape[2])
        pipe = Pipeline([
            ('fbcsp_asp', ImprovedFBCSP_ASP(n_bands=n_bands, n_components=n_comp, n_asp_components=2, n_particles=30, max_iter=20, k_features=16)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        pipe.fit(X_tr, y_tr)
        return pipe.predict(X_te)

class Config4_AddAugmentation(BaseAblationStrategy):
    @property
    def name(self) -> str: return "4_FBCSP-ASP (Smart Channels, With Aug)"
    @property
    def use_smart_channels(self) -> bool: return True
    @property
    def use_augmentation(self) -> bool: return True

    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        n_comp = min(4, X_tr_aug.shape[2])
        pipe = Pipeline([
            ('fbcsp_asp', ImprovedFBCSP_ASP(n_bands=n_bands, n_components=n_comp, n_asp_components=2, n_particles=30, max_iter=20, k_features=16)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        pipe.fit(X_tr_aug, y_tr_aug)
        return pipe.predict(X_te)

class Config5_RiemannBranch(BaseAblationStrategy):
    @property
    def name(self) -> str: return "5_FBRiemann (Smart Channels, No Aug)"
    @property
    def use_smart_channels(self) -> bool: return True
    @property
    def use_augmentation(self) -> bool: return False

    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        pipe = Pipeline([
            ('fbriemann', StandardFBRiemann(n_bands=n_bands)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        # Note: Riemann branch operates on pristine (non-augmented) data to maximize its robustness trait
        pipe.fit(X_tr, y_tr)
        return pipe.predict(X_te)

class Config6_FullEnsemble(BaseAblationStrategy):
    @property
    def name(self) -> str: return "6_Ensemble_Full (FBCSP-ASP+Aug & Riemann+NoAug)"
    @property
    def use_smart_channels(self) -> bool: return True
    @property
    def use_augmentation(self) -> bool: return True

    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        n_comp = min(4, X_tr_aug.shape[2])

        pipe_fbcsp = Pipeline([
            ('fbcsp', ImprovedFBCSP_ASP(n_bands=n_bands, n_components=n_comp, n_asp_components=2, n_particles=30, max_iter=20, k_features=16)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])

        pipe_riemann = Pipeline([
            ('fbriemann', StandardFBRiemann(n_bands=n_bands)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])

        pipe_fbcsp.fit(X_tr_aug, y_tr_aug)
        pipe_riemann.fit(X_tr, y_tr)

        proba_fbcsp = pipe_fbcsp.predict_proba(X_te)
        proba_riemann = pipe_riemann.predict_proba(X_te)

        mean_proba = (proba_fbcsp + proba_riemann) / 2.0
        classes = pipe_fbcsp.classes_
        return classes[np.argmax(mean_proba, axis=1)]

class Config7_FullEnsemble_BaseChannels(BaseAblationStrategy):
    """The Ultimate Control: Full Ensemble without the Smart Channel routing."""
    @property
    def name(self) -> str: return "7_Ensemble_Full (Base Channels, With Aug)"

    @property
    def use_smart_channels(self) -> bool: return False

    @property
    def use_augmentation(self) -> bool: return True

    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        n_comp = min(4, X_tr_aug.shape[2])

        pipe_fbcsp = Pipeline([
            ('fbcsp', ImprovedFBCSP_ASP(n_bands=n_bands, n_components=n_comp, n_asp_components=2, n_particles=30, max_iter=20, k_features=16)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])

        pipe_riemann = Pipeline([
            ('fbriemann', StandardFBRiemann(n_bands=n_bands)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])

        pipe_fbcsp.fit(X_tr_aug, y_tr_aug)
        pipe_riemann.fit(X_tr, y_tr)

        proba_fbcsp = pipe_fbcsp.predict_proba(X_te)
        proba_riemann = pipe_riemann.predict_proba(X_te)

        mean_proba = (proba_fbcsp + proba_riemann) / 2.0
        classes = pipe_fbcsp.classes_
        return classes[np.argmax(mean_proba, axis=1)]

class Config8_SymmetricEnsemble_Aug(BaseAblationStrategy):
    """
    The Routing Control: Symmetric Fusion.
    Both FBCSP-ASP and Riemannian branches receive Augmented data.
    """
    @property
    def name(self) -> str: return "8_Ensemble_Symmetric (Both Branches Augmented)"

    @property
    def use_smart_channels(self) -> bool: return True

    @property
    def use_augmentation(self) -> bool: return True

    def fit_predict(self, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_tr_aug: np.ndarray, y_tr_aug: np.ndarray,
                    X_te: np.ndarray, n_bands: int) -> np.ndarray:
        n_comp = min(4, X_tr_aug.shape[2])

        pipe_fbcsp = Pipeline([
            ('fbcsp', ImprovedFBCSP_ASP(n_bands=n_bands, n_components=n_comp, n_asp_components=2, n_particles=30, max_iter=20, k_features=16)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])

        pipe_riemann = Pipeline([
            ('fbriemann', StandardFBRiemann(n_bands=n_bands)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])

        # THE CRITICAL DIFFERENCE: Both branches now receive the Augmented data
        pipe_fbcsp.fit(X_tr_aug, y_tr_aug)
        pipe_riemann.fit(X_tr_aug, y_tr_aug)

        proba_fbcsp = pipe_fbcsp.predict_proba(X_te)
        proba_riemann = pipe_riemann.predict_proba(X_te)

        mean_proba = (proba_fbcsp + proba_riemann) / 2.0
        classes = pipe_fbcsp.classes_
        return classes[np.argmax(mean_proba, axis=1)]
# ==========================================================
# 6. پردازش سیگنال و بارگذاری داده‌های EEG
# ==========================================================
class EEGDataPreprocessor:
    def __init__(self, bands: List[Tuple[float, float]], tmin: float = 0.5, tmax: float = 3.0):
        self.bands = bands
        self.tmin = tmin
        self.tmax = tmax

    def process_edf_file(self, edf_file: str, chosen_channels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        raw_base = mne.io.read_raw_edf(edf_file, preload=True, eog=['HEOL', 'HEOR'], verbose=False)
        raw_base.rename_channels({'HEOL': 'EOG1', 'HEOR': 'EOG2'}, verbose=False)
        raw_base.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog'}, verbose=False)
        raw_base.pick(chosen_channels, verbose=False)

        try:
            raw_base.set_montage('standard_1005', match_case=False, verbose=False)
            raw_spatial = mne.preprocessing.compute_current_source_density(raw_base, verbose=False)
        except Exception:
            raw_spatial = raw_base.copy().set_eeg_reference(ref_channels='average', verbose=False)

        raw_spatial.resample(500, verbose=False)

        events_df = pd.read_csv("data/task-motor-imagery_events.csv")
        motor_imagery_df = events_df[events_df['value'] == 2]
        onset = motor_imagery_df['onset'] / 1000.0
        duration = motor_imagery_df['duration'] / 1000.0
        description = ['left_hand' if r['trial_type'] == 1 else 'right_hand' for _, r in motor_imagery_df.iterrows()]
        annotations = mne.Annotations(onset=onset, duration=duration, description=description)

        fb_epochs = []
        y_labels = None

        for (l_freq, h_freq) in self.bands:
            raw_filt = raw_spatial.copy().filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_design='firwin2', verbose=False)
            raw_filt.set_annotations(annotations, verbose=False)
            events, event_id = mne.events_from_annotations(raw_filt, event_id={'left_hand': 1, 'right_hand': 2}, verbose=False)

            epochs = mne.Epochs(raw_filt, events, event_id=event_id, tmin=-0.5, tmax=4.0, baseline=(-0.5, 0), preload=True, verbose=False)
            time_mask = (epochs.times >= self.tmin) & (epochs.times <= self.tmax)
            fb_epochs.append(epochs.get_data(copy=False)[:, :, time_mask])

            if y_labels is None:
                y_labels = epochs.events[:, -1]

        X_fb = np.stack(fb_epochs, axis=1)
        return X_fb, y_labels

# ==========================================================
# 7. بدنه اصلی اجرای برنامه (Ablation Evaluation Harness)
# ==========================================================
def main():
    set_seeds(128)

    participants_path = "data/participants.csv"
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"❌ فایل یافت نشد! مطمئن شوید '{participants_path}' در مسیر اجرای کد است.")

    df_part = pd.read_csv(participants_path)

    id_col = next((c for c in df_part.columns if 'id' in c.lower() or 'sub' in c.lower()), df_part.columns[0])
    loc_col = next((c for c in df_part.columns if any(k in c.lower() for k in ['location', 'lesion', 'stroke', 'description'])), df_part.columns[7])
    para_col = next((c for c in df_part.columns if 'paralysis' in c.lower() or 'side' in c.lower()), None)

    BANDS = [(2.0, 4.0), (4.0, 6.0), (6.0, 10.0), (10.0, 14.0), (14.0, 18.0), (18.0, 24.0), (24.0, 30.0)]
    preprocessor = EEGDataPreprocessor(bands=BANDS, tmin=0.5, tmax=3.0)

    subjects = list(range(1, 51))

    ablation_strategies: List[BaseAblationStrategy] = [
        Config1_Baseline(),
        Config2_AddChannelSelection(),
        Config3_AddASP(),
        Config4_AddAugmentation(),
        Config5_RiemannBranch(),
        Config7_FullEnsemble_BaseChannels(), # Novelty Proof 1: Proves Smart Channels work
        Config8_SymmetricEnsemble_Aug(),     # Novelty Proof 2: Proves Asymmetric Routing works
        Config6_FullEnsemble()               # Your Final Proposed Model
    ]
    metrics_list = ['Acc', 'Kappa', 'Sens', 'Prec']

    print(f"🚀 Starting Step-by-Step Ablation Proof Study...")
    for strat in ablation_strategies:
        print(f"   • {strat.name}")
    print()

    all_metric_cols = [f"{strat.name}_{met}" for strat in ablation_strategies for met in metrics_list]
    master_results = {col: [] for col in all_metric_cols}
    master_results['Subject'] = []

    for sub_num in subjects:
        subject_id = f"{sub_num:02d}"
        sub_str_look = f"sub-{subject_id}"

        sub_row = df_part[df_part[id_col].astype(str).str.contains(sub_str_look, case=False, na=False)]

        if not sub_row.empty:
            stroke_loc = sub_row[loc_col].values[0]
            paralysis_side = sub_row[para_col].values[0] if para_col else "Unknown"
        else:
            stroke_loc, paralysis_side = "Unknown", "Unknown"

        smart_channels = ClinicalChannelSelector.get_smart_channels(stroke_loc, paralysis_side)
        base_channels = ClinicalChannelSelector.BASE_CHANNELS

        print(f"⚡ Processing {sub_str_look} | Location: {str(stroke_loc):<30} | Side: {str(paralysis_side):<5}")

        edf_file = f"/media/amirmahdi/NewVolume/edffile/sub-{subject_id}/eeg/sub-{subject_id}_task-motor-imagery_eeg.edf"
        if not os.path.exists(edf_file):
            print(f"[WARNING] File not found for sub-{subject_id}")
            master_results['Subject'].append(sub_str_look)
            for col in all_metric_cols:
                master_results[col].append(np.nan)
            continue

        try:
            X_fb_base, y_base = preprocessor.process_edf_file(edf_file, base_channels)
            X_fb_smart, y_smart = preprocessor.process_edf_file(edf_file, smart_channels)

            cv = StratifiedShuffleSplit(n_splits=10, train_size=0.6, random_state=128)
            splits = list(cv.split(X_fb_smart, y_smart))

            loc = stroke_loc.lower() if isinstance(stroke_loc, str) else ""
            if 'pons' in loc or 'medulla' in loc or 'cerebell' in loc:
                actual_method, dyn_aug_factor = 'Identity', 2
            else:
                actual_method, dyn_aug_factor = 'AVG+NS', 1

            augmenter = AdaptiveClinicalAugmenter(method_str=actual_method, aug_factor=dyn_aug_factor, seed=128)

            strat_fold_metrics: Dict[str, Dict[str, List[float]]] = {
                strat.name: {'Acc': [], 'Kappa': [], 'Sens': [], 'Prec': []} for strat in ablation_strategies
            }

            for train_idx, test_idx in splits:
                # ==========================================
                # 1. Base Channel Data Prep
                # ==========================================
                X_tr_base, y_tr_base = X_fb_base[train_idx], y_base[train_idx]
                X_te_base = X_fb_base[test_idx]

                # NEW: Augment the Base Channel data!
                X_tr_base_aug, y_tr_base_aug = augmenter.augment(X_tr_base, y_tr_base)

                # ==========================================
                # 2. Smart Channel Data Prep
                # ==========================================
                X_tr_smart, y_tr_smart = X_fb_smart[train_idx], y_smart[train_idx]
                X_te_smart, y_te = X_fb_smart[test_idx], y_smart[test_idx]

                # Augment the Smart Channel data
                X_tr_smart_aug, y_tr_smart_aug = augmenter.augment(X_tr_smart, y_tr_smart)

                # ==========================================
                # 3. Strategy Execution
                # ==========================================
                for strat in ablation_strategies:
                    # Dynamically route the correct data streams based on the strategy
                    if strat.use_smart_channels:
                        X_tr_used, y_tr_used = X_tr_smart, y_tr_smart
                        X_tr_aug_used, y_tr_aug_used = X_tr_smart_aug, y_tr_smart_aug
                        X_te_used = X_te_smart
                    else:
                        X_tr_used, y_tr_used = X_tr_base, y_tr_base
                        X_tr_aug_used, y_tr_aug_used = X_tr_base_aug, y_tr_base_aug
                        X_te_used = X_te_base

                    preds = strat.fit_predict(
                        X_tr=X_tr_used,
                        y_tr=y_tr_used,
                        X_tr_aug=X_tr_aug_used,
                        y_tr_aug=y_tr_aug_used,
                        X_te=X_te_used,
                        n_bands=len(BANDS)
                    )

                    # Save Metrics
                    strat_fold_metrics[strat.name]['Acc'].append(accuracy_score(y_te, preds))
                    strat_fold_metrics[strat.name]['Kappa'].append(cohen_kappa_score(y_te, preds))
                    strat_fold_metrics[strat.name]['Sens'].append(recall_score(y_te, preds, pos_label=1, zero_division=0))
                    strat_fold_metrics[strat.name]['Prec'].append(precision_score(y_te, preds, pos_label=1, zero_division=0))

            master_results['Subject'].append(sub_str_look)
            for strat in ablation_strategies:
                for met in metrics_list:
                    mean_val = np.mean(strat_fold_metrics[strat.name][met])
                    master_results[f"{strat.name}_{met}"].append(mean_val)

                acc_m = np.mean(strat_fold_metrics[strat.name]['Acc'])
                print(f"   ↳ [{strat.name:<40}] Mean Acc: {acc_m:.2%}")

        except Exception as e:
            print(f"[ERROR] Fault processing sub-{subject_id}: {e}")
            master_results['Subject'].append(sub_str_look)
            for col in all_metric_cols:
                master_results[col].append(np.nan)

    df_results = pd.DataFrame(master_results)
    df_results.to_csv("ablation_study_logical_proof.csv", index=False)

    print("\n==========================================================================")
    print("📊 ABLATION STUDY GLOBAL SUMMARY")
    print("==========================================================================")

    summary_rows = []
    for strat in ablation_strategies:
        row = {
            "Ablation Mode": strat.name,
            "Accuracy": f"{df_results[f'{strat.name}_Acc'].mean(skipna=True):.2%}",
            "Kappa": f"{df_results[f'{strat.name}_Kappa'].mean(skipna=True):.3f}",
            "Sensitivity": f"{df_results[f'{strat.name}_Sens'].mean(skipna=True):.2%}",
            "Precision": f"{df_results[f'{strat.name}_Prec'].mean(skipna=True):.2%}"
        }
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    print("==========================================================================")
    print("✅ Results saved to 'ablation_study_logical_proof.csv'")

if __name__ == "__main__":
    main()