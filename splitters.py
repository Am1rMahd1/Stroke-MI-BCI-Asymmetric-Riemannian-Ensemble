from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, LeaveOneGroupOut, GroupKFold, \
    GroupShuffleSplit
from interfaces import BaseSplitter

class IntraKFoldSplitter(BaseSplitter):
    def __init__(self, n_splits, random_state, shuffle=False):
        self.cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
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
    def __init__(self):
        self.cv = LeaveOneGroupOut()
    def get_splits(self, X, y, groups):
        return self.cv.split(X, y, groups)

class InterGroupShuffleSplit(BaseSplitter):
    def __init__(self, n_splits, train_size, random_state):
        self.cv = GroupShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)
    def get_splits(self, X, y, groups):
        return self.cv.split(X, y, groups)