# splitters.py
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, LeaveOneGroupOut, GroupKFold
from interfaces import BaseSplitter

class IntraKFoldSplitter(BaseSplitter):
    def __init__(self, config, shuffle=True):
        self.cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=shuffle, random_state=config.RANDOM_STATE if shuffle else None)
    def get_splits(self, X, y, groups=None):
        return self.cv.split(X, y)

class IntraShuffleSplitter(BaseSplitter):
    def __init__(self, config):
        self.cv = StratifiedShuffleSplit(n_splits=config.N_SPLITS, train_size=config.SHUFFLE_TRAIN_SIZE, random_state=config.RANDOM_STATE)
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