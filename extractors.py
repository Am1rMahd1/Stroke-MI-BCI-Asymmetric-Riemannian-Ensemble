# extractors.py
import numpy as np
import mne
from mne.decoding import CSP
from scipy.linalg import fractional_matrix_power

from interfaces import BaseExtractor

class CSPExtractor(BaseExtractor):
    def __init__(self, targetClasses, numberOfComponents, cspRegularization):
        self.targetClasses = targetClasses
        self.numberOfComponents = numberOfComponents
        self.cspRegularization = cspRegularization
        self.csp_model = None

    def _get_clean_data(self, epochs):
        y = epochs.events[:, -1]
        mask = np.isin(y, self.targetClasses)
        return epochs.get_data(copy=False)[mask], y[mask]

    def fit(self, epochs):
        x_raw, y = self._get_clean_data(epochs)

        self.csp_model = CSP(
            n_components=self.numberOfComponents,
            reg=self.cspRegularization,
            log=True,
            norm_trace=True
        )

        self.csp_model.fit(x_raw, y)
        return self

    def transform(self, epochs):
        if self.csp_model is None:
            raise RuntimeError("Call the fit method first.")

        x_raw, y = self._get_clean_data(epochs)

        x_csp = self.csp_model.transform(x_raw)

        return x_csp, y


class FBCSPExtractor(BaseExtractor):
    def __init__(self, targetClasses, freqBands, numberOfComponents, cspRegularization):
        self.targetClasses = targetClasses
        self.freqBands = freqBands
        self.numberOfComponents = numberOfComponents
        self.cspRegularization = cspRegularization
        self.csp_models = {}

    def _get_clean_data(self, epochs):
        y = epochs.events[:, -1]
        mask = np.isin(y, self.targetClasses)
        return epochs.get_data(copy=False)[mask], y[mask], epochs.info['sfreq']

    def fit(self, epochs):
        x_raw, y, sfreq = self._get_clean_data(epochs)

        for band_name, (l_freq, h_freq) in self.freqBands.items():
            x_band = mne.filter.filter_data(
                x_raw, sfreq, l_freq, h_freq, verbose=False
            )

            csp = CSP(
                n_components=self.numberOfComponents,
                reg=self.cspRegularization,
                log=True,
                norm_trace=True
            )
            csp.fit(x_band, y)

            self.csp_models[band_name] = csp

        return self

    def transform(self, epochs):
        x_raw, y, sfreq = self._get_clean_data(epochs)
        x_fbcsp = []

        for band_name, (l_freq, h_freq) in self.freqBands.items():
            x_band = mne.filter.filter_data(
                x_raw, sfreq, l_freq, h_freq, verbose=False
            )

            if band_name in self.csp_models:
                x_band_csp = self.csp_models[band_name].transform(x_band)
                x_fbcsp.append(x_band_csp)
            else:
                raise RuntimeError(f"CSP model is has not been fitted for the band {band_name} .")

        x_fbcsp_concatenated = np.hstack(x_fbcsp)
        return x_fbcsp_concatenated, y