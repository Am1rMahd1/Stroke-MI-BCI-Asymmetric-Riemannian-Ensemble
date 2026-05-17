# import mne
import pandas as pd
from pathlib import Path
from interfaces import BasePreprocessor, BaseLoader

class StrokeEEGPreprocessor(BasePreprocessor):
    def __init__(self, referenceChannels, dropChannels, filter, montageName):
        self.referenceChannels = referenceChannels
        self.dropChannels = dropChannels
        self.filter = filter
        self.montageName = montageName

    def process(self, raw):
        # Channel mapping and typing
        raw.rename_channels({'HEOL' : 'EOG1', 'HEOR' : 'EOG2'}, verbose=False)
        raw.set_channel_types({'EOG1' : 'eog', 'EOG2' : 'eog'}, verbose=False)

        # Referencing and picking
        if self.referenceChannels:  # Only apply if not an empty list
            raw.set_eeg_reference(
                ref_channels=self.referenceChannels,
                verbose=False
            )

        raw.pick_types(
            eeg=True,
            eog=False,
            meg=False,
            stim=False,
            exclude='bads',
            verbose=False
        )

        # raw.pick_channels(
        #     # ch_names=['C3', 'C4', 'Cz', 'CP3', 'CP4', 'CPz'],
        #     ch_names=['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4'],
        #     # ch_names=['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4'],
        #     verbose=False
        # )

        # Dropping specific channels
        raw.drop_channels(
            ch_names=self.dropChannels,
            on_missing='ignore'
        )

        # Filtering
        if self.filter is not None:
            raw.filter(
                l_freq=self.filter["filter_l_freq"],
                h_freq=self.filter["filter_h_freq"],
                method='iir',
                iir_params=self.filter["filter_iir_params"],
                verbose=False,
            )

        # Montage
        montage = mne.channels.make_standard_montage(
            kind=self.montageName
        )
        raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)

        return raw


class SubjectDataLoader(BaseLoader):
    def __init__(self, baseEdfFilesPath, eventsFilePath, preprocessor, epochTMin, epochTMax, resmapleFreq):
        self.baseEdfFilesPath = baseEdfFilesPath
        self.eventsFilePath = eventsFilePath
        self.preprocessor = preprocessor
        self.epochTMin = epochTMin
        self.epochTMax = epochTMax
        self.resmapleFreq = resmapleFreq

    def _extract_epochs(self, raw):
        events_df = pd.read_csv(self.eventsFilePath)
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
            tmin=self.epochTMin,
            tmax=self.epochTMax,
            baseline=None,
            preload=True,
            proj=True,
            verbose=False,
        )

        if self.resmapleFreq != 500:
            epochs.resample(self.resmapleFreq, verbose=False)
        return epochs

    def load_subject_data(self, subject_id):
        edf_file = f"{self.baseEdfFilesPath}/sub-{subject_id:02d}/eeg/sub-{subject_id:02d}_task-motor-imagery_eeg.edf"

        if not Path(edf_file).exists():
            raise FileNotFoundError(f"File {edf_file} not found.")

        raw = mne.io.read_raw_edf(edf_file, preload=True, eog=['HEOL', 'HEOR'], stim_channel=[''], verbose=False)

        # Apply preprocessing
        raw = self.preprocessor.process(raw)

        # Extract epochs
        epochs = self._extract_epochs(raw)

        return epochs