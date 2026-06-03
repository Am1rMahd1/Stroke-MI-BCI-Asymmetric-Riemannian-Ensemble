import numpy as np

class EEGAugmentor:
    """
    Handles data augmentation for EEG signals.
    Must be applied ONLY to training data after splitting.
    """
    def __init__(self, noise_level=0.05):
        # noise_level represents the standard deviation of the Gaussian noise
        self.noise_level = noise_level

    def _add_gaussian_noise(self, X):
        """Adds random Gaussian noise to the EEG signals."""
        # X shape: (epochs, channels, timepoints)
        noise = np.random.normal(0, self.noise_level, X.shape)
        return X + noise

    def augment(self, X_train, y_train, groups_train=None):
        """
        Generates augmented data and appends it to the original training data.
        Returns the expanded dataset.
        """
        # Generate augmented signals
        X_noisy = self._add_gaussian_noise(X_train)

        # Concatenate original and augmented data
        X_augmented = np.vstack((X_train, X_noisy))
        y_augmented = np.concatenate((y_train, y_train))

        if groups_train is not None:
            groups_augmented = np.concatenate((groups_train, groups_train))
            return X_augmented, y_augmented, groups_augmented

        return X_augmented, y_augmented

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
