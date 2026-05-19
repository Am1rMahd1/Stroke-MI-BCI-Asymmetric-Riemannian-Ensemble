import glob
import os
import copy
import random

import pandas as pd
import numpy as np
import mlflow
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from EEGAugmentor import EEGAugmentor, EEGAugmentor2
from EEGnet import EEGNetClassifier, EEGNetClassifierMinimized, AutoTuningEEGNet
from classifiers import SVMClassifier, LDAClassifier, WeightedEnsembleClassifier, KNNClassifier, \
    AutoTuningSVMClassifier, AutoTuningLDAClassifier, AutoTuningKNNClassifier, AutoTuningXGBoostClassifier, \
    AutoTuningRandomForest, AutoTuningLogisticRegression
from config.config import Config
from config.model_parameters import LDA_PARAMS, LDA_GRID, KNN_PARAMS, SVM_PARAMS, SVM_GRID, LR_GRID, KNN_GRID
from extractors import CSPExtractor, FBCSPExtractor, CSPAndVarianceExtractor, RiemannianExtractor
from generate_features import RCSPExtractor
from splitters import IntraKFoldSplitter, IntraShuffleSplitter, InterLOSOSplitter, InterGroupKFoldSplitter

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

def evaluate_intra_subject(X_all, y_all, groups_all, ext_name, ext_params, classifiers, splitter, run_name):
    """ Calculates metrics per subject, logs global averages, and saves per-subject details as an artifact. """

    EXTRACTOR_CLASS_MAP = {
        'FBCSP': FBCSPExtractor,
        'CSP': CSPExtractor,
        'CSPAndVariance': CSPAndVarianceExtractor,
        'RiemannianExtractor': RiemannianExtractor,
        'RCSPExtractor': RCSPExtractor,
    }
    ExtractorClass = EXTRACTOR_CLASS_MAP[ext_name]

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("evaluation_phase", "Intra-Subject")
        mlflow.log_param("validation_strategy", type(splitter).__name__)
        mlflow.log_param("feature_extractor", ext_name)

        for clf in classifiers:
            with mlflow.start_run(run_name=clf.name, nested=True):
                mlflow.log_param("classifier", clf.name)
                mlflow.log_param("is_hyperparameter_tuned", getattr(clf, 'is_tuned', False))

                subject_avg_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }

                # NEW: List to hold the detailed breakdown for the CSV
                detailed_subject_results = []

                unique_subjects = np.unique(groups_all)
                for subject_id in tqdm(unique_subjects, desc=f"Evaluating {clf.name} (Intra-Subject)"):
                    mask = (groups_all == subject_id)
                    X_sub, y_sub = X_all[mask], y_all[mask]

                    fold_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }

                    for train_idx, test_idx in splitter.get_splits(X_sub, y_sub):
                        X_train_raw, y_train = X_sub[train_idx], y_sub[train_idx]
                        X_test_raw, y_test = X_sub[test_idx], y_sub[test_idx]


                        # ==========================================
                        # 1. ADD AUGMENTATION HERE (ONLY ON TRAIN)
                        # ==========================================
                        # augmentor = EEGAugmentor(noise_level=0.001)
                        augmentor = EEGAugmentor2(method='mixup', multiplier=5)
                        X_train_raw, y_train = augmentor.augment(X_train_raw, y_train)
                        # Now X_train_raw and y_train are doubled in size
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

                        fold_extractor = ExtractorClass(**ext_params)

                        # 1. Fit extractor ONLY on training data
                        # fold_extractor.fit(X_train_raw, y_train)

                        # 2. Transform both train and test data
                        # X_train_features = fold_extractor.transform(X_train_raw)
                        # X_test_features = fold_extractor.transform(X_test_raw)
                        X_train_features = X_train_raw
                        X_test_features = X_test_raw
                        # 3. Train classifier on features and predict
                        clf.train(X_train_features, y_train)
                        y_pred = clf.predict(X_test_features)
                        # y_pred = clf.predict(X_train_features)

                        metrics = calculate_metrics(y_test, y_pred)
                        # metrics = calculate_metrics(y_train, y_pred)

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
                csv_filename = f"{clf.name}_intra_subject_details.csv"
                df_results.to_csv(csv_filename, index=False)

                # Attach the CSV to the MLflow run, then clean up the local file
                mlflow.log_artifact(csv_filename)
                if os.path.exists(csv_filename): os.remove(csv_filename)

def evaluate_inter_subject(X_all, y_all, groups_all, classifiers, splitter, run_name):
    """ Calculates metrics globally across the Inter-subject folds (LOSO / GroupKFold). """

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("evaluation_phase", "Inter-Subject")
        mlflow.log_param("validation_strategy", type(splitter).__name__)

        for clf in classifiers:
            with mlflow.start_run(run_name=clf.name, nested=True):
                mlflow.log_param("classifier", clf.name)
                mlflow.log_param("is_hyperparameter_tuned", getattr(clf, 'is_tuned', False))

                fold_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }
                splits = list(splitter.get_splits(X_all, y_all, groups_all))

                for train_idx, test_idx in tqdm(splits, desc=f"Evaluating {clf.name} (Inter-Subject)", leave=True):
                    clf.train(X_all[train_idx], y_all[train_idx])
                    y_pred = clf.predict(X_all[test_idx])
                    metrics = calculate_metrics(y_all[test_idx], y_pred)

                    for k, v in metrics.items(): fold_metrics[k].append(v)

                # Average across all folds and log
                for k, v in fold_metrics.items():
                    mlflow.log_metric(f"mean_{k}", np.mean(v))
                    mlflow.log_metric(f"std_{k}", np.std(v)) # اضافه شدن انحراف معیار برای جدول مقاله


def run_phase1_ablation():
    print("\n" + "="*50 + "\n🚀 PHASE 1: ABLATION STUDY (Table I)\n" + "="*50)

    dataset_files = glob.glob("data/processed/dataset_*.npz")

    if not dataset_files:
        print("❌ No dataset files found. Run generate_features.py first.")
        return

    classifiers = [
        # SVMClassifier(SVM_PARAMS)
        # AutoTuningSVMClassifier(SVM_GRID)
        # KNNClassifier(KNN_PARAMS),
        # AutoTuningKNNClassifier(KNN_GRID),
        # AutoTuningLogisticRegression(LR_GRID)
        # LDAClassifier(LDA_PARAMS),
        # AutoTuningLDAClassifier(LDA_GRID),
        EEGNetClassifier(nb_classes=2,chans=30, samples=512),
        # EEGNetClassifierMinimized(nb_classes=2,chans=9, samples=512),
        # AutoTuningEEGNet(nb_classes=2,chans=9, samples=512),
    ]

    # cv_splitter = IntraShuffleSplitter(
    #     n_splits=config.N_SPLITS,
    #     train_size=config.SHUFFLE_TRAIN_SIZE,
    #     random_state=config.RANDOM_STATE
    # )

    cv_splitter = IntraKFoldSplitter()

    for file_path in dataset_files:
        file_name = os.path.basename(file_path).replace(".npz", "")
        print(f"\n▶️ Evaluating: {file_name}")

        run_params = {
            "dataset_name": file_name
        }

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
            'RCSPExtractor': {},
            # "CSPAndVariance": {
            #     "numberOfComponents": config.N_COMPONENTS,
            #     "regularization": None
            # },
            # "RiemannianExtractor": {
            #     "estimator": 'oas'
            # }
        }
        # extractors = {
        #     # "CSP": CSPExtractor(
        #     #     numberOfComponents=config.N_COMPONENTS,
        #     #     regularization=None
        #     # ),
        #     "FBCSP": FBCSPExtractor(
        #         sfreq=sfreq,
        #         freq_bands=config.FREQ_BANDS,
        #         numberOfComponents=config.N_COMPONENTS,
        #         regularization=None,
        #     )
        # }

        for ext_name, ext_params in extractors.items():
            print(f"  --> Using Extractor: {ext_name}")
            run_name = f"Abl_{file_name}_{ext_name}"

            evaluate_intra_subject(
                X_all=X_all,
                y_all=y_all,
                groups_all=groups_all,
                ext_name=ext_name,       # Pass the string name
                ext_params=ext_params,   # Pass your parameter dictionary
                classifiers=classifiers,
                splitter=cv_splitter,
                run_name=run_name,
            )

def get_golden_data(best_dataset_name):
    data = np.load(f"data/processed/{best_dataset_name}.npz")
    return data['x_data'], data['y_labels'], data['groups']

def run_phase2_intra_models(best_dataset_name):
    print("\n" + "="*50 + "\n🚀 PHASE 2: INTRA-SUBJECT MODEL COMPETITION\n" + "="*50)
    X, y, groups = get_golden_data(best_dataset_name)

    cfg = Config()

    models = [
        SVMClassifier(config),
        AutoTuningSVMClassifier(config),
        LDAClassifier(config),
        AutoTuningLDAClassifier(config),
        WeightedEnsembleClassifier(config),
        KNNClassifier(config),
        AutoTuningKNNClassifier(config),
        AutoTuningXGBoostClassifier(config),
        AutoTuningRandomForest(config),
        AutoTuningLogisticRegression(config),
    ]

    print("Running Shuffled CV...")
    evaluate_intra_subject(X, y, groups, models, IntraShuffleSplitter(cfg), "Phase2_Shuffled", cfg)

    print("Running 10-Fold CV...")
    evaluate_intra_subject(X, y, groups, models, IntraKFoldSplitter(cfg), "Phase2_10Fold", cfg)

def run_phase3_inter_models(base_config):
    print("\n" + "="*50 + "\n🚀 PHASE 3: INTER-SUBJECT (0-SHOT) COMPETITION\n" + "="*50)
    X, y, groups, cfg = get_golden_data(base_config)

    # مدل‌های سبک (Linears & Trees) برای جلوگیری از اورفیت
    models = [
        AutoTuningLDAClassifier(cfg),
        AutoTuningLogisticRegression(cfg),
        AutoTuningRandomForest(cfg) # جایگزین LightGBM در صورت عدم وجود
    ]

    print("Running LOSO (Leave-One-Subject-Out)...")
    evaluate_inter_subject(X, y, groups, models, InterLOSOSplitter(cfg), "Phase3_LOSO", cfg)

    print("Running Group K-Fold...")
    evaluate_inter_subject(X, y, groups, models, InterGroupKFoldSplitter(cfg), "Phase3_GroupKFold", cfg)

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

    run_phase1_ablation()

    # run_phase2_intra_models('features_FBCSP_IIR-4.0-45.0_Ref-Cz_Res-128.0_Reg-None')

    # run_phase3_inter_models(config)

    print("\n✅ All evaluations complete. Run 'mlflow ui' to view your publication-ready results!")