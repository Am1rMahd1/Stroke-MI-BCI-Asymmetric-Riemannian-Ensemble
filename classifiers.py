# classifiers.py
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

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

class AutoTuningSVMClassifier(BaseClassifier):
    def __init__(self, config):
        self.config = config

        # The base model establishes the safety nets: probabilities and balancing
        self.base_model = SVC(
            class_weight='balanced',
            probability=True,
            random_state=config.RANDOM_STATE
        )

        # The GridSearch automatically finds the best C and Kernel internally
        self.model = GridSearchCV(
            estimator=self.base_model,
            param_grid=config.SVM_GRID,
            cv=3,          # 3-fold internal validation for speed
            n_jobs=-1,     # Use all your CPU cores
            verbose=0
        )

    @property
    def name(self):
        return "Auto_Tuned_SVM"

    def train(self, X, y):
        # This automatically searches the grid, then fits the best model on all X and y
        self.model.fit(X, y)

    def predict(self, X):
        # Uses the best estimator found during training
        return self.model.predict(X)

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

class AutoTuningKNNClassifier(BaseClassifier):
    def __init__(self, config):
        self.config = config
        self.base_model = KNeighborsClassifier()
        self.model = GridSearchCV(
            estimator=self.base_model,
            param_grid=config.KNN_GRID,
            cv=3,
            n_jobs=-1,
            verbose=0
        )

    @property
    def name(self):
        return "Auto_Tuned_KNN"

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class AutoTuningLDAClassifier(BaseClassifier):
    def __init__(self, config):
        self.config = config
        self.base_model = LinearDiscriminantAnalysis()
        self.model = GridSearchCV(
            estimator=self.base_model,
            param_grid=config.LDA_GRID,
            cv=3,
            n_jobs=-1,
            verbose=0
        )

    @property
    def name(self):
        return "Auto_Tuned_LDA"

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class AutoTuningXGBoostClassifier(BaseClassifier):
    def __init__(self, config):
        self.config = config
        # eval_metric='logloss' prevents a deprecation warning in newer XGBoost versions
        self.base_model = XGBClassifier(
            eval_metric='logloss',
            random_state=config.RANDOM_STATE
        )
        self.model = GridSearchCV(
            estimator=self.base_model,
            param_grid=config.XGB_GRID,
            cv=3,
            n_jobs=-1,
            verbose=0
        )

    @property
    def name(self):
        return "Auto_Tuned_XGBoost"

    def train(self, X, y):
        # XGBoost requires labels to start at 0. Map [1, 2] to [0, 1]
        y_mapped = y - 1
        self.model.fit(X, y_mapped)

    def predict(self, X):
        # Map the [0, 1] predictions back to your original [1, 2] format
        raw_predictions = self.model.predict(X)
        return raw_predictions + 1

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from interfaces import BaseClassifier

class AutoTuningRandomForest(BaseClassifier):
    def __init__(self, config):
        self.config = config
        self.base_model = RandomForestClassifier(random_state=config.RANDOM_STATE)
        self.model = GridSearchCV(
            estimator=self.base_model,
            param_grid=config.RF_GRID,
            cv=3,
            n_jobs=-1,
            verbose=0
        )

    @property
    def name(self):
        return "Auto_Tuned_RF"

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class AutoTuningLogisticRegression(BaseClassifier):
    def __init__(self, config):
        self.config = config
        # saga solver is required for ElasticNet penalty
        self.base_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=5000,  # High iterations required for saga convergence
            class_weight='balanced',
            random_state=config.RANDOM_STATE
        )
        self.model = GridSearchCV(
            estimator=self.base_model,
            param_grid=config.LR_GRID,
            cv=3,
            n_jobs=-1,
            verbose=0
        )

    @property
    def name(self):
        return "Auto_Tuned_LogReg"

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)