import glob
import os
import random

import pandas as pd
import numpy as np
import mlflow
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from EEGAugmentor import EEGAugmentor2
from EEGnet import EEGNetClassifier
from classifiers import AutoTuningKNNClassifier, SVMClassifier
from config.config import Config
from config.model_parameters import KNN_GRID, SVM_PARAMS
from extractors import FBCSPExtractor
from splitters import IntraKFoldSplitter, InterLOSOSplitter, InterGroupShuffleSplit

def calculate_metrics(y_true, y_pred):
    # 1. Calculate Custom BCI Kappa (pe hardcoded to 0.5)
    p0 = accuracy_score(y_true, y_pred)
    pe = 0.5
    custom_kappa = (p0 - pe) / (1.0 - pe)

    # 2. Generate the Confusion Matrix
    # We explicitly define labels=[1, 2] so Class 1 is always at index 0
    # and Class 2 is always at index 1, preventing any accidental axis flipping.
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])

    # In scikit-learn: Rows = True Labels, Columns = Predicted Labels
    TP = cm[0, 0]  # True Class 1, Predicted Class 1 (MATLAB's confusion_matrix(1,1))
    FN = cm[0, 1]  # True Class 1, Predicted Class 2 (MATLAB's confusion_matrix(2,1))
    FP = cm[1, 0]  # True Class 2, Predicted Class 1 (MATLAB's confusion_matrix(1,2))

    # 3. Calculate Class 1 Metrics (with zero-division protection)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    return {
        "accuracy": p0,
        "precision": precision,
        "sensitivity": sensitivity,
        "kappa": custom_kappa
    }
def evaluate_intra_subject(X_all, y_all, groups_all, ext_name, splitter, run_name):
    """ Calculates metrics per subject, logs global averages, and saves per-subject details as an artifact. """

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("evaluation_phase", "Intra-Subject")
        mlflow.log_param("validation_strategy", type(splitter).__name__)
        mlflow.log_param("feature_extractor", ext_name)

        with mlflow.start_run(run_name="clf.name", nested=True):
            mlflow.log_param("classifier", "clf.name")

            subject_avg_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }

            # NEW: List to hold the detailed breakdown for the CSV
            detailed_subject_results = []

            unique_subjects = np.unique(groups_all)
            for subject_id in tqdm(unique_subjects, desc=f"Evaluating {"clf.name"} (Intra-Subject)"):

                mask = (groups_all == subject_id)
                X_sub, y_sub = X_all[mask], y_all[mask]

                fold_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }

                for train_idx, test_idx in splitter.get_splits(X_sub, y_sub):
                    clf = AutoTuningKNNClassifier(KNN_GRID)
                    # clf = EEGNetClassifier(nb_classes=2,chans=30, samples=897)
                    # clf = TorchDeepLearningClassifier(input_dim=4, epochs=150, batch_size=16, lr=0.005)
                    # clf = TorchRawEEGClassifier(channels=30, samples=1025, epochs=50, batch_size=16)


                    X_train_raw, y_train = X_sub[train_idx], y_sub[train_idx]
                    X_test_raw, y_test = X_sub[test_idx], y_sub[test_idx]

                    # ==========================================
                    # 1.5. ADD Z-NORMALIZATION HERE
                    # ==========================================
                    # Calculate mean/std ONLY on the augmented training data
                    # Assuming X shape is (Samples, Channels, Time)
                    # We calculate mean/std per channel
                    # train_mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
                    # train_std = X_train_raw.std(axis=(0, 2), keepdims=True)
                    #
                    # # Add a small epsilon to avoid division by zero
                    # train_std[train_std == 0] = 1e-8
                    #
                    # # Normalize training and testing sets
                    # X_train_raw = (X_train_raw - train_mean) / train_std
                    # X_test_raw = (X_test_raw - train_mean) / train_std


                    # ==========================================
                    # 1. ADD AUGMENTATION HERE (ONLY ON TRAIN)
                    # ==========================================
                    # augmentor = EEGAugmentor(noise_level=0.001)
                    augmentor = EEGAugmentor2(method='mixup', multiplier=5)
                    X_train_raw, y_train = augmentor.augment(X_train_raw, y_train)
                    # Now X_train_raw and y_train are doubled in size
                    # ==========================================

                    # 1. Fit extractor ONLY on training data
                    # fold_extractor = FBCSPExtractor(256, config.FREQ_BANDS, config.N_COMPONENTS, 'ledoit_wolf')
                    # fold_extractor.fit(X_train_raw, y_train)

                    # 2. Transform both train and test data
                    # X_train_features = fold_extractor.transform(X_train_raw)
                    # X_test_features = fold_extractor.transform(X_test_raw)
                    X_train_features = X_train_raw
                    X_test_features = X_test_raw

                    # 3. Train classifier on features and predict
                    clf.train(X_train_features, y_train)
                    y_pred = clf.predict(X_test_features)

                    metrics = calculate_metrics(y_test, y_pred)

                    for k, v in metrics.items(): fold_metrics[k].append(v)

                # Average folds for THIS subject
                sub_dict = {"subject_id": subject_id}
                for k in subject_avg_metrics.keys():
                    sub_mean = np.mean(fold_metrics[k])
                    subject_avg_metrics[k].append(sub_mean)
                    sub_dict[k] = sub_mean # Store for the CSV

                detailed_subject_results.append(sub_dict)

            # 1. Log the global averages across ALL subjects to the main UI
            for k, v in subject_avg_metrics.items():
                mlflow.log_metric(f"mean_{k}", np.mean(v))
                mlflow.log_metric(f"std_{k}", np.std(v))

            # 2. Save the detailed per-subject breakdown as a CSV artifact
            df_results = pd.DataFrame(detailed_subject_results)
            csv_filename = f"{"clf.name"}_intra_subject_details.csv"
            df_results.to_csv(csv_filename, index=False)

            # Attach the CSV to the MLflow run, then clean up the local file
            mlflow.log_artifact(csv_filename)
            if os.path.exists(csv_filename): os.remove(csv_filename)
def evaluate_inter_subject(X_all, y_all, groups_all, ext_name, classifiers, splitter, run_name):
    """ Calculates metrics globally across the Inter-subject folds (LOSO / GroupKFold). """

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("evaluation_phase", "Inter-Subject")
        mlflow.log_param("validation_strategy", type(splitter).__name__)
        mlflow.log_param("feature_extractor", ext_name)

        for clf in classifiers:
            with mlflow.start_run(run_name=clf.name, nested=True):
                mlflow.log_param("classifier", clf.name)
                mlflow.log_param("is_hyperparameter_tuned", getattr(clf, 'is_tuned', False))

                fold_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }
                splits = list(splitter.get_splits(X_all, y_all, groups_all))

                for train_idx, test_idx in tqdm(splits, desc=f"Evaluating {clf.name} (Inter-Subject)", leave=True):
                    X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
                    X_test_raw, y_test = X_all[test_idx], y_all[test_idx]

                    # ==========================================
                    # 1. ADD AUGMENTATION HERE (ONLY ON TRAIN)
                    # ==========================================
                    augmentor = EEGAugmentor2(method='time_shift', multiplier=3)
                    X_train_raw, y_train = augmentor.augment(X_train_raw, y_train)
                    # ==========================================

                    # ==========================================
                    # 1.5. ADD Z-NORMALIZATION HERE
                    # ==========================================
                    # Calculate mean/std ONLY on the augmented training data
                    # Assuming X shape is (Samples, Channels, Time)
                    # We calculate mean/std per channel
                    train_mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
                    train_std = X_train_raw.std(axis=(0, 2), keepdims=True)
                    # Add a small epsilon to avoid division by zero
                    train_std[train_std == 0] = 1e-8
                    # Normalize training and testing sets
                    X_train_raw = (X_train_raw - train_mean) / train_std
                    X_test_raw = (X_test_raw - train_mean) / train_std

                    X_train_features = X_train_raw
                    X_test_features = X_test_raw

                    # 3. Train classifier on features and predict
                    clf.train(X_train_features, y_train)
                    y_pred = clf.predict(X_test_features)

                    metrics = calculate_metrics(y_test, y_pred)

                    for k, v in metrics.items(): fold_metrics[k].append(v)

                # Average across all folds and log
                for k, v in fold_metrics.items():
                    mlflow.log_metric(f"mean_{k}", np.mean(v))
                    mlflow.log_metric(f"std_{k}", np.std(v)) # اضافه شدن انحراف معیار برای جدول مقاله
def evaluate_inter_subject_with_meta(X_all, y_all, groups_all, metadata_csv_path, classifiers, splitter, run_name):
    """
    X_all: (trials, channels, samples)
    groups_all: (trials,) -> Contains participant IDs (e.g., 'sub-01')
    metadata_csv_path: Path to the preprocessed CSV
    """
    # 1. Load and Map Metadata
    meta_df = pd.read_csv(metadata_csv_path)
    # ایجاد یک دیکشنری برای دسترسی سریع: {'sub-01': [features], ...}
    # فرض بر این است که ستون اول فایل CSV همان ID است، اگر نیست نام ستون را جایگزین کنید
    meta_lookup = {i+1: meta_df.iloc[i].values for i in range(len(meta_df))}

    with mlflow.start_run(run_name=run_name):
        for clf in classifiers:
            with mlflow.start_run(run_name=clf.name, nested=True):
                fold_metrics = {"accuracy": [], "precision": [], "sensitivity": [], "kappa": []}
                splits = list(splitter.get_splits(X_all, y_all, groups_all))

                for train_idx, test_idx in tqdm(splits, desc=f"LOSO - {clf.name}"):
                    # جداسازی EEG
                    X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
                    X_test_raw, y_test = X_all[test_idx], y_all[test_idx]

                    # استخراج متادیتا متناظر با سوژه‌های هر Trial
                    meta_train = np.array([meta_lookup[s] for s in groups_all[train_idx]])
                    meta_test = np.array([meta_lookup[s] for s in groups_all[test_idx]])

                    # --- Augmentation ---
                    multiplier = 3
                    augmentor = EEGAugmentor2(method='time_shift', multiplier=multiplier)
                    X_train_raw, y_train = augmentor.augment(X_train_raw, y_train)
                    # تکرار متادیتا برای داده‌های Augment شده
                    meta_train = np.tile(meta_train, (multiplier, 1))

                    # ==========================================
                    # 1.5. ADD Z-NORMALIZATION HERE
                    # ==========================================
                    # Calculate mean/std ONLY on the augmented training data
                    # Assuming X shape is (Samples, Channels, Time)
                    # We calculate mean/std per channel
                    train_mean = X_train_raw.mean(axis=(0, 2), keepdims=True)
                    train_std = X_train_raw.std(axis=(0, 2), keepdims=True)
                    # Add a small epsilon to avoid division by zero
                    train_std[train_std == 0] = 1e-8
                    # Normalize training and testing sets
                    X_train_raw = (X_train_raw - train_mean) / train_std
                    X_test_raw = (X_test_raw - train_mean) / train_std

                    X_train_features = X_train_raw
                    X_test_features = X_test_raw

                    # --- Train & Predict ---
                    clf.train(X_train_features, meta_train, y_train)
                    y_pred = clf.predict(X_test_features, meta_test)

                    # Calculate Metrics
                    metrics = calculate_metrics(y_test, y_pred)

                    for k, v in metrics.items(): fold_metrics[k].append(v)

                # Average across all folds and log
                for k, v in fold_metrics.items():
                    mlflow.log_metric(f"mean_{k}", np.mean(v))
                    mlflow.log_metric(f"std_{k}", np.std(v))
def run_phase1_intra_subject():
    print("\n" + "="*50 + "\n🚀 PHASE 1: Intra Subject (Table I)\n" + "="*50)

    dataset_files = glob.glob("data/processed/dataset_*.npz")

    if not dataset_files:
        print("❌ No dataset files found. Run generate_features.py first.")
        return

    for file_path in dataset_files:
        file_name = os.path.basename(file_path).replace(".npz", "")
        print(f"\n▶️ Evaluating: {file_name}")

        config = Config()

        data = np.load(file_path)

        X_all = data['x_data']
        y_all = data['y_labels']
        groups_all = data['groups']
        sfreq = data['samplingFreq']

        extractors = {
            # 'FBCSP': {
            #     'sfreq': sfreq,
            #     'freq_bands': config.FREQ_BANDS,
            #     "numberOfComponents": config.N_COMPONENTS,
            #     "regularization": None
            # },
            'None': {},
            # "CSPAndVariance": {
            #     "numberOfComponents": config.N_COMPONENTS,
            #     "regularization": None
            # },
            # "RiemannianExtractor": {
            #     "estimator": 'oas'
            # }
        }

        for ext_name, ext_params in extractors.items():
            print(f"  --> Using Extractor: {ext_name}")
            run_name = f"Abl_{file_name}_{ext_name}"

            evaluate_intra_subject(
                X_all=X_all,
                y_all=y_all,
                groups_all=groups_all,
                ext_name=ext_name,       # Pass the string name
                splitter=IntraKFoldSplitter(),
                run_name=run_name,
            )

            # evaluate_intra_subject(
            #     X_all=X_all,
            #     y_all=y_all,
            #     groups_all=groups_all,
            #     ext_name=ext_name,       # Pass the string name
            #     splitter=IntraShuffleSplitter(
            #         n_splits=config.N_SPLITS,
            #         train_size=config.SHUFFLE_TRAIN_SIZE,
            #         random_state=config.RANDOM_STATE
            #     ),
            #     run_name=run_name,
            # )
def get_golden_data(best_dataset_name):
    data = np.load(f"data/processed/{best_dataset_name}.npz")
    return data['x_data'], data['y_labels'], data['groups']
def run_phase2_inter_models():
    print("\n" + "="*50 + "\n🚀 PHASE 2: INTER-SUBJECT COMPETITION\n" + "="*50)
    X, y, groups = get_golden_data('dataset_step_50')

    models = [
        # AutoTuningLDAClassifier(cfg),
        # AutoTuningLogisticRegression(cfg),
        # AutoTuningRandomForest(cfg)
        EEGNetClassifier(nb_classes=2,chans=30, samples=512),
    ]

    print("Running LOSO (Leave-One-Subject-Out)...")
    evaluate_inter_subject(
        X_all=X,
        y_all=y,
        groups_all=groups,
        ext_name="None",
        ext_params={},
        classifiers=models,
        splitter=InterLOSOSplitter(),
        run_name="Phase3_LOSO",
    )

    print("Running Group K-Fold...")
    evaluate_inter_subject(
        X_all=X,
        y_all=y,
        groups_all=groups,
        ext_name="None",
        ext_params={},
        classifiers=models,
        splitter=InterGroupShuffleSplit(config.N_SPLITS),
        run_name="Phase3_GroupKFold",
    )
def run_phase3_inter_models_with_metadata():
    print("\n" + "="*50 + "\n🚀 PHASE 3: INTER-SUBJECT WITH METADATA\n" + "="*50)
    X, y, groups = get_golden_data('dataset_step_50')

    models = [
        EEGNetClassifier(nb_classes=2,chans=30, samples=512),
    ]

    print("Running LOSO with metadata...")
    evaluate_inter_subject_with_meta(
        X_all=X,
        y_all=y,
        groups_all=groups,
        ext_name="None",
        ext_params={},
        classifiers=models,
        splitter=InterLOSOSplitter(),
        metadata_csv_path='meta/preprocessed.csv',
        run_name="Phase3_LOSO_WITH_METADATA",
    )

    print("Running Group K-Fold...")
    evaluate_inter_subject(
        X_all=X,
        y_all=y,
        groups_all=groups,
        ext_name="None",
        ext_params={},
        classifiers=models,
        splitter=InterGroupShuffleSplit(config.N_SPLITS),
        run_name="Phase3_GroupKFold",
    )
def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    # Python random module
    random.seed(seed)
    # Python environment
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # PyTorch CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    config = Config()
    set_seed(config.RANDOM_STATE)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    run_phase1_intra_subject()

    # run_phase2_inter_models()

    # run_phase3_inter_models_with_metadata()

    print("\n✅ All evaluations complete. Run 'mlflow ui' to view your publication-ready results!")