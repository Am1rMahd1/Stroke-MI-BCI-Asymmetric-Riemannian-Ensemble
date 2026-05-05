import mne
import pandas as pd
from pathlib import Path
from interfaces import BasePreprocessor, BaseLoader

class StrokeEEGPreprocessor(BasePreprocessor):
    def __init__(self, config):
        self.config = config

    def process(self, raw):
        # Channel mapping and typing
        raw.rename_channels({'HEOL' : 'EOG1', 'HEOR' : 'EOG2'}, verbose=False)
        raw.set_channel_types({'EOG1' : 'eog', 'EOG2' : 'eog'}, verbose=False)

        # Referencing and picking
        raw.set_eeg_reference(self.config.REFERENCE_CHANNELS, verbose=False)
        raw.pick_types(eeg=True, eog=False, meg=False, stim=False, exclude='bads', verbose=False)

        # Dropping specific channels
        raw.drop_channels(self.config.DROP_CHANNELS, on_missing='ignore')

        # Filtering
        if self.config.L_FREQ is not None and self.config.H_FREQ is not None:
            raw.filter(
                l_freq=self.config.L_FREQ,
                h_freq=self.config.H_FREQ,
                method='iir',
                iir_params=self.config.IIR_PARAMS,
                verbose=False,
            )

        # Montage
        montage = mne.channels.make_standard_montage(self.config.MONTAGE_NAME)
        raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)

        return raw


class SubjectDataLoader(BaseLoader):
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor

    def _extract_epochs(self, raw):
        events_df = pd.read_csv(self.config.EVENTS_FILE)
        motor_imagery_df = events_df[events_df['value'] == 2]

        onset = motor_imagery_df['onset'] / 1000.0
        duration = motor_imagery_df['duration'] / 1000.0

        description = []
        for _, row in motor_imagery_df.iterrows():
            trial_type = row['trial_type']
            if trial_type == 1:
                description.append('left_hand')
            else:
                description.append('right_hand')

        annotations = mne.Annotations(onset=onset, duration=duration, description=description)
        raw.set_annotations(annotations, verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=self.config.EPOCH_TMIN,
            tmax=self.config.EPOCH_TMAX,
            baseline=None,
            preload=True,
            proj=True,
            verbose=False,
        )

        if self.config.FS_RESAMPLE is not None:
            epochs.resample(self.config.FS_RESAMPLE, verbose=False)
        return epochs

    def load_subject_data(self, subject_id):
        edf_file = f"{self.config.BASE_EDF_PATH}/sub-{subject_id:02d}/eeg/sub-{subject_id:02d}_task-motor-imagery_eeg.edf"

        if not Path(edf_file).exists():
            raise FileNotFoundError(f"File {edf_file} not found.")

        raw = mne.io.read_raw_edf(edf_file, preload=True, eog=['HEOL', 'HEOR'], stim_channel=[''], verbose=False)

        # Apply preprocessing
        raw = self.preprocessor.process(raw)

        # Extract epochs
        epochs = self._extract_epochs(raw)

        return epochs