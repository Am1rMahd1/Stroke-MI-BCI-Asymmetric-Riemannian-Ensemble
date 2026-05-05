# config.py

class Config:
    # Paths
    BASE_EDF_PATH = "/media/amirmahdi/NewVolume/edffile"
    EVENTS_FILE = "data/task-motor-imagery_events.csv"

    # Active Pipeline Toggle
    # Change this to 'CSP' or 'FBCSP' to automatically adjust the pipeline and filename
    EXTRACTOR_TYPE = 'FBCSP'

    @property
    def OUTPUT_FILE(self):
        iir_str = f"IIR-{self.L_FREQ}-{self.H_FREQ}" if self.L_FREQ is not None else "IIR-False"
        res_str = f"Res-{self.FS_RESAMPLE}" if self.FS_RESAMPLE is not None else "Res-False"
        reg_str = f"Reg-{self.CSP_REGULARIZATION}" if hasattr(self, 'CSP_REGULARIZATION') else "Reg-None"

        if self.REFERENCE_CHANNELS == 'average':
            ref_str = "Ref-CAR"
        elif isinstance(self.REFERENCE_CHANNELS, list) and len(self.REFERENCE_CHANNELS) > 0:
            ref_str = f"Ref-{self.REFERENCE_CHANNELS[0]}" # تبدیل می‌شود به Ref-Cz
        else:
            ref_str = "Ref-None"

        filename = f"features_{self.EXTRACTOR_TYPE}_{iir_str}_{ref_str}_{res_str}_{reg_str}.npz"
        return f"data/processed/{filename}"

    # Preprocessing
    FS_RESAMPLE = 128.0
    L_FREQ = 4.0
    H_FREQ = 45.0
    IIR_PARAMS = dict(order=2, ftype='butter')
    REFERENCE_CHANNELS = ['Cz']
    DROP_CHANNELS = ['']
    MONTAGE_NAME = 'standard_1020'

    # Epoching
    EPOCH_TMIN = 0.0
    EPOCH_TMAX = 4.0
    TARGET_CLASSES = [1, 2]  # 1: left_hand, 2: right_hand

    # Feature Extraction
    N_COMPONENTS = 12
    FREQ_BANDS = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0),
    }

    # --- Data Tracking ---
    PROCESSED_DATA_FILE = "data/processed/features_FBCSP_4.0_45.0Hz.npz"
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = "Rayaparto_BCI_Stroke_Classification"

    # --- Cross Validation Parameters ---
    N_SPLITS = 10
    SHUFFLE_TRAIN_SIZE = 0.6
    RANDOM_STATE = 36  # Kept your specific random state

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