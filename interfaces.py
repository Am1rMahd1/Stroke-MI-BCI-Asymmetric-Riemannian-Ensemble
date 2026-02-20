# interfaces.py
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, raw):
        pass

class BaseLoader(ABC):
    @abstractmethod
    def load_subject_data(self, subject_id):
        pass

class BaseExtractor(ABC):
    @abstractmethod
    def fit_transform(self, epochs):
        pass

class BaseSplitter(ABC):
    @abstractmethod
    def get_splits(self, x_data, y_labels, groups):
        pass

class BaseClassifier(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass