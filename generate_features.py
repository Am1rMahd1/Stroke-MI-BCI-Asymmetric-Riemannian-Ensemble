import os
import numpy as np
from config.config import Config
from data_loader import StrokeEEGPreprocessor, SubjectDataLoader


def generate_and_save_dataset(step_config):
    config = Config()

    preprocessor = StrokeEEGPreprocessor(
        referenceChannels=step_config["ref"],
        dropChannels=config.DROP_CHANNELS,
        filter=step_config["filter"],
        montageName=config.MONTAGE_NAME,
    )
    loader = SubjectDataLoader(
        baseEdfFilesPath=config.BASE_EDF_PATH,
        eventsFilePath=config.EVENTS_FILE,
        preprocessor=preprocessor,
        epochTMin=config.EPOCH_TMIN,
        epochTMax=config.EPOCH_TMAX,
        resmapleFreq=step_config["resample"],
    )

    all_x = []
    all_y = []
    all_groups = []

    output_filename = f"dataset_step_{step_config["name"]}.npz"
    output_path = os.path.join(config.PROCESSED_DATA_DIR, output_filename)

    # if os.path.exists(output_path):
    #     print(f"Dataset already exists at {output_path}. Skipping.")
    #     return output_path

    for subject_id in range(1, 6):
        # 1. (Filtering, Resampling, Referencing)
        epochs = loader.load_subject_data(subject_id)
        y_labels = epochs.events[:, -1]
        mask = np.isin(y_labels, config.TARGET_CLASSES)

        x_data = epochs.get_data(copy=False)[mask]
        y_labels = y_labels[mask]

        groups = np.full(len(y_labels), subject_id)
        all_x.append(x_data)
        all_y.append(y_labels)
        all_groups.append(groups)

        print(f"Processed Subject {subject_id} - Signal Shape: {x_data.shape}")

    # ذخیره دیتاست (این بار سیگنال‌ها ذخیره می‌شوند نه ویژگی‌های نهایی)
    np.savez_compressed(
        output_path,
        x_data=np.vstack(all_x),
        y_labels=np.concatenate(all_y),
        groups=np.concatenate(all_groups),
        samplingFreq=step_config["resample"],
    )
    return output_path

if __name__ == "__main__":

    ablation_steps = [
        # {
        #     "step": 1,
        #     "name": "1",
        #     "filter": None,
        #     "ref": [],
        #     "resample": 500,
        #     # "reg": None
        # },
        # {
        #     "step": 2,
        #     "name": "2",
        #     "filter": {
        #         "filter_l_freq": 4.0,
        #         "filter_h_freq": 45.0,
        #         "filter_iir_params": dict(order=2, ftype='butter'),
        #     },
        #     "ref": [],
        #     "resample": 500,
        #     # "reg": None
        # },
        # {
        #     "step": 3,
        #     "name": "3",
        #     "filter": {
        #         "filter_l_freq": 4.0,
        #         "filter_h_freq": 45.0,
        #         "filter_iir_params": dict(order=2, ftype='butter'),
        #     },
        #     "ref": 'average',
        #     "resample": 500,
        #     # "reg": None
        # },
        # {
        #     "step": 4,
        #     "name": "4",
        #     "filter": {
        #         "filter_l_freq": 4.0,
        #         "filter_h_freq": 45.0,
        #         "filter_iir_params": dict(order=2, ftype='butter'),
        #     },
        #     "ref": ['Cz'],
        #     "resample": 500,
        #     # "reg": None
        # },
        # {
        #     "step": 5,
        #     "name": "5",
        #     "filter": {
        #         "filter_l_freq": 4.0,
        #         "filter_h_freq": 45.0,
        #         "filter_iir_params": dict(order=2, ftype='butter'),
        #     },
        #     "ref": ['Cz'],
        #     "resample": 128.0,
        #     # "reg": None
        # },
        {
            "step": 6,
            "name": "6",
            "filter": {
                "filter_l_freq": 4.0,
                "filter_h_freq": 45.0,
                "filter_iir_params": dict(order=4, ftype='butter'),
            },
            "ref": ['Cz'],
            "resample": 128.0,
            # "reg": None
        },
    ]

    print("🚀 Starting Automated Dataset Generation for Ablation Study...")

    for condition in ablation_steps:
        print(f"\n{'='*60}")
        print(f"▶️ Executing Step {condition['step']}: {condition['name']}")
        print(f"{'='*60}")

        generate_and_save_dataset(condition)

    print("\n🎉 All 7 datasets have been successfully generated and saved!")