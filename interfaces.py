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