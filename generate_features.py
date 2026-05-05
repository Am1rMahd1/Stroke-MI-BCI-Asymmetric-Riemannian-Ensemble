import os
import numpy as np
from config import Config
from data_loader import StrokeEEGPreprocessor, SubjectDataLoader
from extractors import CSPExtractor, FBCSPExtractor, EAFBCSPExtractor

def generate_and_save_dataset(config):
    preprocessor = StrokeEEGPreprocessor(config)
    loader = SubjectDataLoader(config, preprocessor)

    # Dynamically select the extractor based on config
    if config.EXTRACTOR_TYPE == 'FBCSP':
        extractor = FBCSPExtractor(config)
    elif config.EXTRACTOR_TYPE == 'CSP':
        extractor = CSPExtractor(config)
    elif config.EXTRACTOR_TYPE == 'EA_FBCSP':
        extractor = EAFBCSPExtractor(config)
    else:
        raise ValueError("Invalid EXTRACTOR_TYPE. Choose 'CSP', 'FBCSP' or 'EA_FBCSP'.")

    all_x = []
    all_y = []
    all_groups = []

    print(f"\n--- Starting Extraction: {config.EXTRACTOR_TYPE} ---")
    output_path = config.OUTPUT_FILE

    if os.path.exists(output_path):
        print(f"Dataset already exists at {output_path}. Skipping generation.")
        return output_path

    for subject_id in range(1, 6):
        try:
            epochs = loader.load_subject_data(subject_id)
            x_features, y_labels = extractor.fit_transform(epochs)

            groups = np.full(len(y_labels), subject_id)

            all_x.append(x_features)
            all_y.append(y_labels)
            all_groups.append(groups)

            print(f"Processed Subject {subject_id}")

        except FileNotFoundError:
            print(f"Skipping Subject {subject_id} (Not found)")

    x_stacked = np.vstack(all_x)
    y_stacked = np.concatenate(all_y)
    groups_stacked = np.concatenate(all_groups)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        x_data=x_stacked,
        y_labels=y_stacked,
        groups=groups_stacked
    )
    print(f"✅ Dataset securely saved to {output_path} with shape {x_stacked.shape}")

    return output_path

if __name__ == "__main__":
    import copy

    base_config = Config()

    ablation_steps = [
        # {"step": 1, "name": "1_Raw",       "iir": False,  "ref": None,      "resample": False, "ext": "CSP",      "reg": None},
        # {"step": 2, "name": "2_IIR",       "iir": True,   "ref": None,      "resample": False, "ext": "CSP",      "reg": None},
        # {"step": 3, "name": "3_RefCAR",    "iir": True,   "ref": 'average', "resample": False, "ext": "CSP",      "reg": None}, # حالت جدید!
        # {"step": 4, "name": "4_RefCz",     "iir": True,   "ref": ['Cz'],    "resample": False, "ext": "CSP",      "reg": None}, # ارتقا به CAR
        # {"step": 5, "name": "5_Resample",  "iir": True,   "ref": ['Cz'],    "resample": True,  "ext": "CSP",      "reg": None},
        {"step": 6, "name": "6_FBCSP",     "iir": True,   "ref": ['Cz'],    "resample": True,  "ext": "FBCSP",    "reg": None},
        # {"step": 7, "name": "7_EAFBCSP",   "iir": True,   "ref": ['Cz'],    "resample": True,  "ext": "EA_FBCSP", "reg": "ledoit_wolf"},
    ]

    print("🚀 Starting Automated Dataset Generation for Ablation Study...")

    for condition in ablation_steps:
        print(f"\n{'='*60}")
        print(f"▶️ Executing Step {condition['step']}: {condition['name']}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(base_config)

        cfg.L_FREQ = 4.0 if condition["iir"] else None
        cfg.H_FREQ = 45.0 if condition["iir"] else None

        cfg.REFERENCE_CHANNELS = condition["ref"] if condition["ref"] is not None else []
        cfg.FS_RESAMPLE = 128.0 if condition["resample"] else None

        cfg.EXTRACTOR_TYPE = condition["ext"]
        cfg.CSP_REGULARIZATION = condition["reg"]

        try:
            generate_and_save_dataset(cfg)
        except Exception as e:
            print(f"❌ Error in Step {condition['step']}: {str(e)}")

    print("\n🎉 All 7 datasets have been successfully generated and saved!")