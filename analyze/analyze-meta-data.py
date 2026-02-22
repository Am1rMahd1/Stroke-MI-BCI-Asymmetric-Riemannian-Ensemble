import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

def load_and_preprocess(meta_path, results_path):
    print("Loading and merging datasets...")
    df_meta = pd.read_csv(meta_path)
    df_res = pd.read_csv(results_path)

    # 1. Extract ID and Merge
    df_meta['subject_id'] = df_meta['Participant_ID'].str.extract(r'sub-(\d+)').astype(int)
    df = pd.merge(df_meta, df_res, on='subject_id')

    # 2. Preprocess Stroke Location into Anatomical Categories
    print("Preprocessing clinical features...")
    locations = df['StrokeLocation'].str.lower()

    # Hemisphere
    df['Lesion_Hemisphere'] = 'Unclear'
    df.loc[locations.str.contains('left') & ~locations.str.contains('right'), 'Lesion_Hemisphere'] = 'Left'
    df.loc[locations.str.contains('right') & ~locations.str.contains('left'), 'Lesion_Hemisphere'] = 'Right'
    df.loc[locations.str.contains('bilateral') | (locations.str.contains('left') & locations.str.contains('right')), 'Lesion_Hemisphere'] = 'Bilateral'

    # Brain Regions (Binary)
    df['Is_Brainstem'] = locations.str.contains('pons|medulla').astype(int)
    df['Is_Subcortical'] = locations.str.contains('basal ganglia|thalamus|corona radiata|internal capsule|centrum semiovale').astype(int)
    df['Is_Cortical'] = locations.str.contains('frontal|parietal|occipital|temporal|cortex').astype(int)

    # 3. Preprocess Paralysis vs Handedness match
    df['Dominant_Hand_Affected'] = (df['ParalysisSide'] == df['Handedness']).astype(int)

    # Save the cleaned dataset for your records
    df.to_csv("analyze/results/preprocessed_clinical_metadata.csv", index=False)
    print("Saved merged data to 'preprocessed_clinical_metadata.csv'")

    return df

def analyze_continuous_features(df):
    print("\nRunning Correlation Analysis (Continuous Features)...")
    continuous_cols = ['Age', 'Duration', 'NIHSS', 'MBI', 'mRS', 'accuracy', 'kappa']

    # Use Spearman for clinical ordinal data
    corr_matrix = df[continuous_cols].corr(method='spearman')

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Spearman Correlation: Clinical Metadata vs BCI Performance')
    plt.tight_layout()
    plt.savefig('analyze/results/correlation_heatmap.png', dpi=300)
    plt.close()
    print("Saved 'correlation_heatmap.png'")

    # Scatter plot for NIHSS
    plt.figure(figsize=(6, 5))
    sns.regplot(x='NIHSS', y='accuracy', data=df, scatter_kws={'s':80, 'alpha':0.6}, line_kws={'color':'red'})
    plt.title('Stroke Severity (NIHSS) vs BCI Accuracy')
    plt.xlabel('NIHSS Score (Higher = More Severe)')
    plt.ylabel('Classification Accuracy')
    plt.tight_layout()
    plt.savefig('analyze/results/nihss_vs_accuracy.png', dpi=300)
    plt.close()
    print("Saved 'nihss_vs_accuracy.png'")

def test_binary_category(df, col_name, group1_val, group0_val):
    g1 = df[df[col_name] == group1_val]['accuracy']
    g0 = df[df[col_name] == group0_val]['accuracy']

    if len(g1) > 0 and len(g0) > 0:
        stat, p = mannwhitneyu(g1, g0, alternative='two-sided')
        return {
            'Clinical Feature': col_name,
            'Group 1': group1_val,
            'Group 2': group0_val,
            'p-value': round(p, 4),
            'G1 Mean Acc': round(g1.mean(), 3),
            'G2 Mean Acc': round(g0.mean(), 3)
        }
    return None

def analyze_categorical_features(df):
    print("\nRunning Mann-Whitney U Tests (Categorical Features)...")

    tests = [
        test_binary_category(df, 'Is_Brainstem', 1, 0),
        test_binary_category(df, 'Is_Cortical', 1, 0),
        test_binary_category(df, 'Is_Subcortical', 1, 0),
        test_binary_category(df, 'Dominant_Hand_Affected', 1, 0),
        test_binary_category(df, 'Gender', 'male', 'female'),
        test_binary_category(df, 'IsFirstTime', 'yes', 'no')
    ]

    df_stats = pd.DataFrame([t for t in tests if t is not None])
    df_stats = df_stats.sort_values('p-value')

    # Save statistical results to CSV
    df_stats.to_csv("analyze/results/statistical_test_results.csv", index=False)
    print("Saved test results to 'statistical_test_results.csv'")
    print("\n=== CLINICAL SIGNIFICANCE RESULTS ===")
    print(df_stats.to_string(index=False))

    # Visualization
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.boxplot(x='Is_Brainstem', y='accuracy', data=df, ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title('Accuracy: Brainstem vs Other Lesions', fontsize=14)
    axes[0, 0].set_xticklabels(['Other Lesions (0)', 'Brainstem Lesions (1)'])

    sns.boxplot(x='Lesion_Hemisphere', y='accuracy', data=df, ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title('Accuracy by Lesion Hemisphere', fontsize=14)

    sns.boxplot(x='Is_Cortical', y='accuracy', data=df, ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title('Accuracy: Cortical vs Other Lesions', fontsize=14)
    axes[1, 0].set_xticklabels(['Other Lesions (0)', 'Cortical Lesions (1)'])

    sns.boxplot(x='Dominant_Hand_Affected', y='accuracy', data=df, ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title('Accuracy: Dominant Hand Paralysis', fontsize=14)
    axes[1, 1].set_xticklabels(['No (0)', 'Yes (1)'])

    plt.tight_layout()
    plt.savefig('analyze/results/clinical_analysis_plots.png', dpi=300)
    plt.close()
    print("Saved 'clinical_analysis_plots.png'")

def main():
    # File paths (ensure these match your actual files)
    meta_file = "data/participants.csv"
    results_file = "data/Auto_Tuned_LDA_intra_subject_details.csv"

    # 1. Preprocess
    df = load_and_preprocess(meta_file, results_file)

    # 2. Continuous Analysis (Heatmaps/Scatters)
    analyze_continuous_features(df)

    # 3. Categorical Analysis (Boxplots/Mann-Whitney)
    analyze_categorical_features(df)

    print("\nAll analyses complete! Check your folder for the generated plots and CSVs.")

if __name__ == "__main__":
    main()