# extractors.py
import numpy as np
import mne
from mne.decoding import CSP
from scipy.linalg import fractional_matrix_power

from interfaces import BaseExtractor

class CSPExtractor(BaseExtractor):
    def __init__(self, config):
        self.config = config

    def fit_transform(self, epochs):
        x_epochs = epochs.get_data(copy=False)
        y = epochs.events[:, -1]

        mask = np.isin(y, self.config.TARGET_CLASSES)
        x_epochs = x_epochs[mask]
        y = y[mask]

        csp = CSP(n_components=self.config.N_COMPONENTS, reg=None, log=True, norm_trace=True)
        x_csp = csp.fit_transform(x_epochs, y)

        return x_csp, y


class FBCSPExtractor(BaseExtractor):
    def __init__(self, config):
        self.config = config

    def fit_transform(self, epochs):
        x_epochs = epochs.get_data(copy=False)
        y = epochs.events[:, -1]

        mask = np.isin(y, self.config.TARGET_CLASSES)
        x_epochs = x_epochs[mask]
        y = y[mask]

        x_fbcsp = []

        for band_name, (l_freq, h_freq) in self.config.FREQ_BANDS.items():
            x_band = mne.filter.filter_data(
                x_epochs,
                epochs.info['sfreq'],
                l_freq,
                h_freq,
                verbose=False,
            )

            csp = CSP(n_components=self.config.N_COMPONENTS, reg=None, log=True, norm_trace=True)
            x_band_csp = csp.fit_transform(x_band, y)
            x_fbcsp.append(x_band_csp)

        x_fbcsp_concatenated = np.hstack(x_fbcsp)
        return x_fbcsp_concatenated, y

class EAFBCSPExtractor(BaseExtractor):
    def __init__(self, config):
        self.config = config

    def fit_transform(self, epochs):
        x_epochs = epochs.get_data(copy=False)
        y = epochs.events[:, -1]

        mask = np.isin(y, self.config.TARGET_CLASSES)
        x_epochs = x_epochs[mask]
        y = y[mask]

        # ==========================================
        # 1. EUCLIDEAN ALIGNMENT (EA)
        # ==========================================
        covariances = []
        for epoch in x_epochs:
            cov = np.cov(epoch)
            covariances.append(cov)

        reference_matrix = np.mean(covariances, axis=0)

        # --- THE FIX: Diagonal Regularization ---
        # Nudge the diagonal to fix the CAR filter rank-deficiency
        epsilon = 1e-5 * (np.trace(reference_matrix) / reference_matrix.shape[0])
        reference_matrix += epsilon * np.eye(reference_matrix.shape[0])
        # ----------------------------------------

        R = fractional_matrix_power(reference_matrix, -0.5).real

        x_epochs_aligned = np.array([R @ epoch for epoch in x_epochs])

        # ==========================================
        # 2. FILTER BANK CSP (FBCSP)
        # ==========================================
        x_fbcsp = []

        for band_name, (l_freq, h_freq) in self.config.FREQ_BANDS.items():
            x_band = mne.filter.filter_data(
                x_epochs_aligned,
                epochs.info['sfreq'],
                l_freq,
                h_freq,
                verbose=False,
            )

            # Keep ledoit_wolf here as well!
            csp = CSP(n_components=self.config.N_COMPONENTS, reg='ledoit_wolf', log=True, norm_trace=True)
            x_band_csp = csp.fit_transform(x_band, y)
            x_fbcsp.append(x_band_csp)

        x_fbcsp_concatenated = np.hstack(x_fbcsp)

        return x_fbcsp_concatenated, y