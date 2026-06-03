# --- Model Base Parameters (Avoid defaults in functions) ---
SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 0.1,                 # Lower C = Softer margin (ignores outliers)
    'gamma': 'scale',         # Dynamically scales based on feature variance
    'class_weight': 'balanced',
    'probability': True
}

# The Automated Tuning Grid (For GridSearchCV)
SVM_GRID = {
    'C': [0.01, 0.1, 1, 10],               # Explores Soft to Hard margins
    'kernel': ['linear', 'rbf'],           # Linear often beats RBF in high-dimensional BCI
    'gamma': ['scale', 'auto', 0.01, 0.1]  # Only applies to RBF
}
# KNN Grid
KNN_GRID = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# LDA Grid (Restricted to solvers that support shrinkage)
LDA_GRID = {
    'solver': ['lsqr', 'eigen'],
    'shrinkage': ['auto', 0.1, 0.5, 0.9]
}

# XGBoost Grid
XGB_GRID = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}
# Random Forest Grid
RF_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Logistic Regression Grid (ElasticNet requires the 'saga' solver)
LR_GRID = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9],  # Balances L1 (feature selection) and L2 (smoothness)
}
DT_PARAMS = {'criterion': 'gini', 'max_depth': None}
KNN_PARAMS = {'n_neighbors': 5}
LDA_PARAMS = {'solver': 'svd'}
ADABOOST_PARAMS = {'n_estimators': 50, 'learning_rate': 0.1}
ENSEMBLE_WEIGHTS = [2.0, 1.0, 2.2, 1.5, 0.8]