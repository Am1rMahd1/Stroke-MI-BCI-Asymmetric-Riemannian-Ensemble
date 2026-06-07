# config.py

class Config:
    # Paths
    BASE_EDF_PATH = "/media/amirmahdi/129699CC9699B0A9/edffile"
    EVENTS_FILE = "data/task-motor-imagery_events.csv"
    PROCESSED_DATA_DIR='data/processed/'

    N_SPLITS=10
    SHUFFLE_TRAIN_SIZE=0.6

    # Preprocessing

    DROP_CHANNELS = ['']
    MONTAGE_NAME = 'standard_1020'

    # Epoching
    EPOCH_TMIN = 0.5
    EPOCH_TMAX = 4.0
    TARGET_CLASSES = [1, 2]  # 1: left_hand, 2: right_hand

    # Feature Extraction
    # N_COMPONENTS = 8
    N_COMPONENTS = 4
    # FREQ_BANDS = {
    #     # 'delta': (0.5, 4.0),
    #     'theta': (4.0, 8.0),
    #     'alpha': (8.0, 12.0),
    #     'beta': (13.0, 25.0),
    #     # 'gamma': (30.0, 45.0),
    # }
    FREQ_BANDS = {
        'delta': (10.0, 13.0),
        'theta': (13.0, 16.0),
        'alpha': (16.0, 19.0),
        'beta': (19.0, 22.0),
        'gamma': (22.0, 25.0),
    }

    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    EXPERIMENT_NAME = "BCI_Stroke_Classification_2"

    # --- Cross Validation Parameters ---
    RANDOM_STATE = 128  # Kept your specific random state