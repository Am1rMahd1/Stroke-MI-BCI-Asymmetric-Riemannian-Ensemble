# splitters.py
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, LeaveOneGroupOut, GroupKFold

from config.config import Config
from interfaces import BaseSplitter

class IntraKFoldSplitter(BaseSplitter):
    def __init__(self, shuffle=True):
        self.config = Config()
        self.cv = StratifiedKFold(
            n_splits=self.config.N_SPLITS,
            shuffle=shuffle,
            random_state=self.config.RANDOM_STATE if shuffle else None
        )
    def get_splits(self, X, y, groups=None):
        return self.cv.split(X, y)

class IntraShuffleSplitter(BaseSplitter):
    def __init__(self, n_splits, train_size, random_state):
        self.cv = StratifiedShuffleSplit(
            n_splits=n_splits,
            train_size=train_size,
            random_state=random_state
        )
    def get_splits(self, X, y, groups=None):
        return self.cv.split(X, y)

class InterLOSOSplitter(BaseSplitter):
    def __init__(self, config=None):
        self.cv = LeaveOneGroupOut()
    def get_splits(self, X, y, groups):
        return self.cv.split(X, y, groups)

class InterGroupKFoldSplitter(BaseSplitter):
    def __init__(self, config):
        self.cv = GroupKFold(n_splits=config.N_SPLITS)
    def get_splits(self, X, y, groups):
        return self.cv.split(X, y, groups)