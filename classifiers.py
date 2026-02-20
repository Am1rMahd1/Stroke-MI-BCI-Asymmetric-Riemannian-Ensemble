# classifiers.py
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from interfaces import BaseClassifier

class SVMClassifier(BaseClassifier):
    def __init__(self, config):
        self.model = SVC(**config.SVM_PARAMS, random_state=config.RANDOM_STATE)
    @property
    def name(self): return "SVM"
    def train(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class LDAClassifier(BaseClassifier):
    def __init__(self, config):
        self.model = LinearDiscriminantAnalysis(**config.LDA_PARAMS)
    @property
    def name(self): return "LDA"
    def train(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class WeightedEnsembleClassifier(BaseClassifier):
    def __init__(self, config):
        base_models = [
            ('svm', SVC(**config.SVM_PARAMS, random_state=config.RANDOM_STATE)),
            ('dt', DecisionTreeClassifier(**config.DT_PARAMS, random_state=config.RANDOM_STATE)),
            ('knn', KNeighborsClassifier(**config.KNN_PARAMS)),
            ('nb', GaussianNB()),
            ('lda', LinearDiscriminantAnalysis(**config.LDA_PARAMS)),
        ]
        self.model = VotingClassifier(
            estimators=base_models,
            voting='soft',
            weights=config.ENSEMBLE_WEIGHTS,
            n_jobs=-1
        )
    @property
    def name(self): return "Weighted_Ensemble"
    def train(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)

class KNNClassifier(BaseClassifier):
    def __init__(self, config):
        self.model = KNeighborsClassifier(**config.KNN_PARAMS)
    @property
    def name(self): return "KNN"
    def train(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)
