# extractors.py
import numpy as np
import mne
import scipy
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.linalg import fractional_matrix_power
from sklearn.neighbors import kneighbors_graph

from interfaces import BaseExtractor

class CSPExtractor:
    def __init__(self, numberOfComponents, regularization):
        """
        Standard Common Spatial Pattern (CSP) extractor for NumPy arrays.
        """
        self.name="CSP"
        self.csp = CSP(
            n_components=numberOfComponents,
            reg=regularization,
            log=True,
            norm_trace=True
        )

    def fit(self, X, y):
        """
        Fits the CSP on the training data.
        X shape: (n_epochs, n_channels, n_times)
        y shape: (n_epochs,)
        """
        self.csp.fit(X, y)
        return self

    def transform(self, X):
        """
        Extracts features from the data.
        """
        return self.csp.transform(X)

class FBCSPExtractor:
    def __init__(self, sfreq, freq_bands, numberOfComponents, regularization):
        """
        Filter Bank Common Spatial Pattern (FBCSP) extractor for NumPy arrays.
        sfreq: Float, sampling frequency of the data (loaded from your .npz file)
        freq_bands: List of tuples, e.g., [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
        """
        self.name="FBCSP"
        self.sfreq = sfreq
        self.freq_bands = freq_bands
        self.n_components = numberOfComponents

        # Initialize a separate CSP for each frequency band
        self.csps = [
            CSP(
                n_components=numberOfComponents,
                reg=regularization,
                log=True,
                norm_trace=True,
            )
            for _ in freq_bands
        ]

    def _filter_data(self, X):
        """
        Filters the input array X into multiple frequency bands.
        """
        filtered_X = []
        for l_freq, h_freq in self.freq_bands.values():
            # mne.filter.filter_data handles NumPy arrays perfectly
            X_band = mne.filter.filter_data(
                X,
                sfreq=self.sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                iir_params=dict(order=4, ftype='butter'),
                method='iir',
                verbose=False
            )
            filtered_X.append(X_band)
        return filtered_X

    def fit(self, X, y):
        """
        Filters data into bands and fits a CSP model for each band.
        """
        filtered_X = self._filter_data(X)
        for i, X_band in enumerate(filtered_X):
            self.csps[i].fit(X_band, y)
        return self

    def transform(self, X):
        """
        Filters data, applies the fitted CSPs, and concatenates the features.
        """
        filtered_X = self._filter_data(X)
        features = []
        for i, X_band in enumerate(filtered_X):
            band_features = self.csps[i].transform(X_band)
            features.append(band_features)

        # Concatenate features horizontally (combining all bands for each epoch)
        return np.hstack(features)

class CSPAndVarianceExtractor:
    def __init__(self, numberOfComponents, regularization):
        self.name = "CSP_Plus_Variance"
        self.csp = CSP(n_components=numberOfComponents, reg=regularization, log=True, norm_trace=True)

    def fit(self, X, y):
        self.csp.fit(X, y)
        return self

    def transform(self, X):
        # 1. Get CSP features
        csp_features = self.csp.transform(X)

        # 2. Get Log-Variance of each channel across time
        # X shape is (epochs, channels, time) -> variance along time axis (axis=2)
        var_features = np.log(np.var(X, axis=2) + 1e-8)

        # 3. Concatenate horizontally
        # Result shape: (n_epochs, n_csp_components + n_channels)
        combined_features = np.hstack((csp_features, var_features))

        return combined_features

class RiemannianExtractor:
    def __init__(self, estimator):
        """
        Extracts Tangent Space features from covariance matrices.
        'oas' (Oracle Approximating Shrinkage) is excellent for small sample sizes.
        """
        self.name = "Riemannian_TangentSpace"
        self.cov = Covariances(estimator=estimator)
        self.ts = TangentSpace(metric='riemann')

    def fit(self, X, y):
        """
        X shape: (n_epochs, n_channels, n_times)
        """
        covmats = self.cov.fit_transform(X)
        self.ts.fit(covmats, y)
        return self

    def transform(self, X):
        covmats = self.cov.transform(X)
        return self.ts.transform(covmats)

class LPPExtractor:
    def __init__(self, numberOfComponents=2, numberOfNeighbors=5):
        """
        Locality Preserving Projection (LPP) extractor.
        Accepts 3D raw EEG data arrays directly and extracts features internally
        to maintain pipeline compatibility with 3D spatial filters.
        """
        self.name = "LPP"
        self.numberOfComponents = numberOfComponents
        self.numberOfNeighbors = numberOfNeighbors
        self.projection_matrix_ = None

    def _convert_to_features(self, X):
        """
        Internal helper to compress 3D raw time-series (epochs, channels, times)
        into a 2D log-variance feature matrix (epochs, features).
        If the input is already 2D, it passes through untouched.
        """
        if X.ndim == 3:
            return np.log(np.var(X, axis=-1))
        return X

    def fit(self, X, y=None):
        """
        Fits the LPP on the training data.
        X shape: (n_epochs, n_channels, n_times)
        y shape: (n_epochs,)
        """
        # Automatically transform 3D raw array to 2D log-variance features internally
        X_feat = self._convert_to_features(X)

        n_samples, n_features = X_feat.shape

        # Ensure neighbor count doesn't exceed sample pool boundaries
        k = min(self.numberOfNeighbors, n_samples - 1)

        # 1. Construct the local adjacency graph via K-Nearest Neighbors
        W = kneighbors_graph(X_feat, k, mode='connectivity', include_self=False).toarray()
        W = np.maximum(W, W.T)  # Ensure graph matrix symmetry

        # 2. Compute Degree Matrix (D) and Graph Laplacian (L)
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        # 3. Project data to feature space: Xᵀ · L · X and Xᵀ · D · X
        XTLX = X_feat.T @ L @ X_feat
        XTDX = X_feat.T @ D @ X_feat

        # Add a tiny regularization ridge to prevent mathematical singularity
        XTDX += np.eye(n_features) * 1e-6

        # 4. Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = scipy.linalg.eigh(XTLX, XTDX)

        # Sort eigenvectors by ascending eigenvalues
        sorted_indices = np.argsort(eigenvalues)

        # Save the transformation mapping matrix
        self.projection_matrix_ = eigenvectors[:, sorted_indices[:self.numberOfComponents]]

        return self

    def transform(self, X):
        """
        Extracts features from the raw data and projects them into the LPP manifold.
        X shape: (n_epochs, n_channels, n_times)
        Returns shape: (n_epochs, numberOfComponents)
        """
        if self.projection_matrix_ is None:
            raise ValueError("The LPP instance must be fitted before calling transform.")

        # Automatically transform 3D raw array to 2D log-variance features internally
        X_feat = self._convert_to_features(X)

        return X_feat @ self.projection_matrix_