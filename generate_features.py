import os
import numpy as np
from config.config import Config
from data_loader import StrokeEEGPreprocessor, SubjectDataLoader
from scipy.linalg import eigh


def generate_and_save_dataset(step_config):
    config = Config()

    preprocessor = StrokeEEGPreprocessor(
        referenceChannels=step_config["ref"],
        dropChannels=config.DROP_CHANNELS,
        filter=step_config["filter"],
        montageName=config.MONTAGE_NAME,
    )
    loader = SubjectDataLoader(
        baseEdfFilesPath=config.BASE_EDF_PATH,
        eventsFilePath=config.EVENTS_FILE,
        preprocessor=preprocessor,
        epochTMin=config.EPOCH_TMIN,
        epochTMax=config.EPOCH_TMAX,
        resmapleFreq=step_config["resample"],
    )

    all_x = []
    all_y = []
    all_groups = []

    output_filename = f"dataset_step_{step_config["name"]}.npz"
    output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)

    # if os.path.exists(output_path):
    #     print(f"Dataset already exists at {output_path}. Skipping.")
    #     return output_path

    for subject_id in range(1, 51):
    # for subject_id in [6, 15, 43, 17]:
    # for subject_id in [1, 15, 11, 40, 23, 44]:
        # 1. (Filtering, Resampling, Referencing)
        epochs = loader.load_subject_data(subject_id)
        y_labels = epochs.events[:, -1]
        mask = np.isin(y_labels, config.TARGET_CLASSES)

        x_data = epochs.get_data(copy=False)[mask]
        y_labels = y_labels[mask]

        groups = np.full(len(y_labels), subject_id)
        all_x.append(x_data)
        all_y.append(y_labels)
        all_groups.append(groups)

        print(f"Processed Subject {subject_id} - Signal Shape: {x_data.shape}")

    # ذخیره دیتاست (این بار سیگنال‌ها ذخیره می‌شوند نه ویژگی‌های نهایی)
    np.savez_compressed(
        output_path,
        x_data=np.vstack(all_x),
        y_labels=np.concatenate(all_y),
        groups=np.concatenate(all_groups),
        samplingFreq=step_config["resample"],
    )
    return output_path

if __name__ == "__main__":

    ablation_steps = [
        # {
        #     "step": 1,
        #     "name": "1",
        #     "filter": None,
        #     "ref": [],
        #     "resample": 500,
        #     # "reg": None
        # },
        # {
        #     "step": 2,
        #     "name": "2",
        #     "filter": {
        #         "filter_l_freq": 4.0,
        #         "filter_h_freq": 45.0,
        #         "filter_iir_params": dict(order=2, ftype='butter'),
        #     },
        #     "ref": [],
        #     "resample": 500,
        #     # "reg": None
        # },
        # {
        #     "step": 3,
        #     "name": "3",
        #     "filter": {
        #         "filter_l_freq": 4.0,
        #         "filter_h_freq": 45.0,
        #         "filter_iir_params": dict(order=2, ftype='butter'),
        #     },
        #     "ref": 'average',
        #     "resample": 500,
        #     # "reg": None
        # },
        # {
        #     "step": 4,
        #     "name": "4",
        #     "filter": {
        #         "filter_l_freq": 4.0,
        #         "filter_h_freq": 45.0,
        #         "filter_iir_params": dict(order=2, ftype='butter'),
        #     },
        #     "ref": ['Cz'],
        #     "resample": 500,
        #     # "reg": None
        # },
        # {
        #     "step": 5,
        #     "name": "5",
        #     "filter": {
        #         "filter_l_freq": 4.0,
        #         "filter_h_freq": 45.0,
        #         "filter_iir_params": dict(order=2, ftype='butter'),
        #     },
        #     "ref": ['Cz'],
        #     "resample": 128.0,
        #     # "reg": None
        # },
        {
            "step": 6,
            "name": "6",
            "filter": None,
            # "filter": {
            #     "filter_l_freq": 4.0,
            #     "filter_h_freq": 45.0,
            #     "filter_iir_params": dict(order=2, ftype='butter'),
            # },
            "ref": 'average',
            "resample": 128.0,
            # "reg": None
        },
    ]

    print("🚀 Starting Automated Dataset Generation for Ablation Study...")

    for condition in ablation_steps:
        print(f"\n{'='*60}")
        print(f"▶️ Executing Step {condition['step']}: {condition['name']}")
        print(f"{'='*60}")

        generate_and_save_dataset(condition)

    print("\n🎉 All 7 datasets have been successfully generated and saved!")

class RCSPExtractor:
    def __init__(self, n_components=4, X_generic=None, y_generic=None, beta=0.5):
        self.n_components = n_components
        self.beta = beta
        self.X_generic = X_generic
        self.y_generic = y_generic
        self.filters_ = None

    def _compute_class_covariances(self, X, y):
        """
        Computes the average normalized spatial covariance for each class.
        X shape: (n_epochs, n_channels, n_times)
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP is a 2-class algorithm.")

        covs = []
        for c in classes:
            # Select epochs for this class
            X_c = X[y == c]
            n_epochs, n_channels, n_times = X_c.shape

            # Initialize covariance matrix for this class
            class_cov = np.zeros((n_channels, n_channels))

            for i in range(n_epochs):
                # Standard CSP normalization: (X*X') / trace(X*X')
                epoch = X_c[i]
                cov = np.dot(epoch, epoch.T)
                class_cov += cov / np.trace(cov)

            # Average across epochs
            covs.append(class_cov / n_epochs)

        return covs[0], covs[1]

    def fit(self, X_train, y_train):
        # 1. Compute covariance matrices for the Target Subject
        C_target_0, C_target_1 = self._compute_class_covariances(X_train, y_train)

        # 2. Compute covariance matrices for the Generic Pool
        if self.X_generic is not None and self.y_generic is not None:
            C_gen_0, C_gen_1 = self._compute_class_covariances(self.X_generic, self.y_generic)

            # 3. Apply Regularization (The RCSP Math)
            C_reg_0 = (1 - self.beta) * C_target_0 + self.beta * C_gen_0
            C_reg_1 = (1 - self.beta) * C_target_1 + self.beta * C_gen_1
        else:
            # Fallback to standard CSP if no generic data is provided
            C_reg_0, C_reg_1 = C_target_0, C_target_1

        # 4. Solve the generalized eigenvalue problem: C_reg_0 * W = lambda * (C_reg_0 + C_reg_1) * W
        # This sorts eigenvalues in ascending order
        evals, evecs = eigh(C_reg_0, C_reg_0 + C_reg_1)

        # 5. Select the filters
        # We take the first n/2 (discriminative for class 1)
        # and the last n/2 (discriminative for class 0)
        ix = np.argsort(np.abs(evals - 0.5))[::-1] # Alternative: sort by distance from 0.5

        # Standard approach: take the ends of the spectrum
        half = self.n_components // 2
        selected_indices = np.concatenate([np.arange(half), np.arange(len(evals) - half, len(evals))])

        # Filters are the columns of evecs
        self.filters_ = evecs[:, selected_indices].T

        return self

    def transform(self, X):
        """
        Projects X onto the filters and returns the log-variance features.
        X shape: (n_epochs, n_channels, n_times)
        Returns: (n_epochs, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("The model must be fitted before calling transform.")

        n_epochs = X.shape[0]
        features = np.zeros((n_epochs, self.n_components))

        for i in range(n_epochs):
            # Project data: (n_components, n_channels) * (n_channels, n_times)
            projected = np.dot(self.filters_, X[i])

            # Compute variance along the time axis for each component
            var = np.var(projected, axis=1)

            # Log-transform to normalize the distribution
            features[i, :] = np.log(var)

        return features