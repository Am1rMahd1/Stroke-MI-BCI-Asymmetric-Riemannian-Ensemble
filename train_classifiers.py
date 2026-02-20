# train_classifiers.py
import pandas as pd
import os
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix
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
        mlflow.log_param("evaluation_mode", "Intra-Subject")
        mlflow.log_param("dataset", config.PROCESSED_DATA_FILE)

        for clf in classifiers:
            with mlflow.start_run(run_name=clf.name, nested=True):
                subject_avg_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }

                # NEW: List to hold the detailed breakdown for the CSV
                detailed_subject_results = []

                unique_subjects = np.unique(groups_all)
                for subject_id in unique_subjects:
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

                # 2. Save the detailed per-subject breakdown as a CSV artifact
                df_results = pd.DataFrame(detailed_subject_results)
                csv_filename = f"{clf.name}_intra_subject_details.csv"
                df_results.to_csv(csv_filename, index=False)

                # Attach the CSV to the MLflow run, then clean up the local file
                mlflow.log_artifact(csv_filename)
                os.remove(csv_filename)
def evaluate_inter_subject(X_all, y_all, groups_all, classifiers, splitter, run_name, config):
    """ Calculates metrics globally across the Inter-subject folds (LOSO / GroupKFold). """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("evaluation_mode", "Inter-Subject")
        mlflow.log_param("dataset", config.PROCESSED_DATA_FILE)

        for clf in classifiers:
            with mlflow.start_run(run_name=clf.name, nested=True):
                fold_metrics = { "accuracy": [], "precision": [], "sensitivity": [], "kappa": [] }

                for train_idx, test_idx in splitter.get_splits(X_all, y_all, groups_all):
                    clf.train(X_all[train_idx], y_all[train_idx])
                    y_pred = clf.predict(X_all[test_idx])
                    metrics = calculate_metrics(y_all[test_idx], y_pred)

                    for k, v in metrics.items(): fold_metrics[k].append(v)

                # Average across all folds and log
                for k, v in fold_metrics.items():
                    mlflow.log_metric(f"mean_{k}", np.mean(v))

def main():
    config = Config()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # 1. Load Data generated in Phase 1
    data = np.load(config.PROCESSED_DATA_FILE)
    X, y, groups = data['x_data'], data['y_labels'], data['groups']

    # 2. Define active models
    models = [
        SVMClassifier(config),
        AutoTuningSVMClassifier(config),
        LDAClassifier(config),
        AutoTuningLDAClassifier(config),
        # WeightedEnsembleClassifier(config),
        KNNClassifier(config),
        AutoTuningKNNClassifier(config),
        AutoTuningXGBoostClassifier(config),
        AutoTuningRandomForest(config),
        AutoTuningLogisticRegression(config),

    ]

    # 3. Execute Intra-Subject Phase
    print("Running Intra-Subject (Stratified K-Fold with Shuffle)...")
    evaluate_intra_subject(X, y, groups, models, IntraShuffleSplitter(config), "Intra_KFold_Shuffle with 12 csp components 120 hz resampling with FP1", config)

    # 4. Execute Inter-Subject Phase
    print("Running Inter-Subject (Leave-One-Subject-Out)...")
    # evaluate_inter_subject(X, y, groups, models, InterGroupKFoldSplitter(config), "Inter_LOSO with 12 csp components 120 hz resampling with FP1", config)

    print("Evaluation complete. Run 'mlflow ui' to view results.")

if __name__ == "__main__":
    main()