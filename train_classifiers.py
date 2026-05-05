import glob
import os
import copy
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from config import Config
from classifiers import SVMClassifier, LDAClassifier, WeightedEnsembleClassifier, KNNClassifier, \
    AutoTuningSVMClassifier, AutoTuningLDAClassifier, AutoTuningKNNClassifier, AutoTuningXGBoostClassifier, \
    AutoTuningRandomForest, AutoTuningLogisticRegression
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

def evaluate_intra_subject(X_all, y_all, groups_all, classifiers, splitter, run_name, config):
    """ Calculates metrics per subject, logs global averages, and saves per-subject details as an artifact. """

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("evaluation_phase", "Intra-Subject")
        mlflow.log_param("validation_strategy", type(splitter).__name__)

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
                        clf.train(X_sub[train_idx], y_sub[train_idx])
                        y_pred = clf.predict(X_sub[test_idx])
                        metrics = calculate_metrics(y_sub[test_idx], y_pred)

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

def evaluate_inter_subject(X_all, y_all, groups_all, classifiers, splitter, run_name, config):
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


def run_phase1_ablation(base_config):
    print("\n" + "="*50 + "\n🚀 PHASE 1: ABLATION STUDY (Table I)\n" + "="*50)

    dataset_files = glob.glob("data/processed/features_*.npz")

    if not dataset_files:
        print("❌ No dataset files found. Run generate_features.py first.")
        return

    base_lda = LDAClassifier(base_config)
    auto_tuned_lda = AutoTuningLDAClassifier(base_config)
    cv_splitter = IntraShuffleSplitter(base_config)

    for file_path in dataset_files:
        file_name = os.path.basename(file_path).replace(".npz", "")
        print(f"\n▶️ Evaluating: {file_name}")

        run_params = {
            "dataset_name": file_name
        }

        data = np.load(file_path)

        evaluate_intra_subject(
            data['x_data'],
            data['y_labels'],
            data['groups'],
            [base_lda, auto_tuned_lda],
            cv_splitter,
            f"Abl_{file_name}",
            base_config,
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

if __name__ == "__main__":
    config = Config()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    run_phase1_ablation(config)

    run_phase2_intra_models('features_FBCSP_IIR-4.0-45.0_Ref-Cz_Res-128.0_Reg-None')

    run_phase3_inter_models(config)

    print("\n✅ All evaluations complete. Run 'mlflow ui' to view your publication-ready results!")