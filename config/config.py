# config.py

class Config:
    # Paths
    BASE_EDF_PATH = "/media/amirmahdi/NewVolume/edffile"
    EVENTS_FILE = "data/task-motor-imagery_events.csv"
    PROCESSED_DATA_DIR='data/processed/'

    # Preprocessing

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

    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = "BCI_Stroke_Classification"

    # --- Cross Validation Parameters ---
    RANDOM_STATE = 36  # Kept your specific random state