class Config:
    # Paths
    BASE_EDF_PATH = "/media/amirmahdi/NewVolume/edffile"
    EVENTS_FILE = "data/task-motor-imagery_events.csv"
    OUTPUT_FILE = "data/processed/bci_features.npz"

    # Preprocessing
    FS_RESAMPLE = 250.0
    L_FREQ = 4.0
    H_FREQ = 45.0
    IIR_PARAMS = dict(order=2, ftype='butter')
    REFERENCE_CHANNELS = ['Cz']
    DROP_CHANNELS = ['FP1']
    MONTAGE_NAME = 'standard_1020'

    # Epoching
    EPOCH_TMIN = 0.0
    EPOCH_TMAX = 4.0
    TARGET_CLASSES = [1, 2]  # 1: left_hand, 2: right_hand

    # Feature Extraction
    N_COMPONENTS = 4
    FREQ_BANDS = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 45.0),
    }