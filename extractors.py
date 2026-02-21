# extractors.py
import numpy as np
import mne
from mne.decoding import CSP
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