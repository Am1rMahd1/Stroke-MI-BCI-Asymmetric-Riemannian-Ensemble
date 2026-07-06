import os
import random
import warnings
import numpy as np
import pandas as pd
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==========================================================
# تابع استراتژی هوشمند انتخاب کانال بر اساس نوع سکته
# ==========================================================
def get_smart_channels(location_str):
    if pd.isna(location_str) or not isinstance(location_str, str):
        return ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']

    loc = location_str.lower()
    base_channels = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']

    # قانون ۱: سکته‌های ساقه مغز/مخچه -> تمرکز روی هسته مرکزی برای کاهش نویز عضلانی شدید
    if 'pons' in loc or 'medulla' in loc or 'cerebell' in loc:
        return ['FCz', 'C3', 'Cz', 'C4', 'CPz']

    smart_channels = base_channels.copy()

    # قانون ۲: سکته‌های نیمکره چپ
    if 'left' in loc:
        if 'frontal' in loc or 'cortex' in loc or 'radiata' in loc:
            smart_channels += ['F3', 'Fz']
        if 'parietal' in loc or 'subparietal' in loc:
            smart_channels += ['P3', 'Pz']

    # قانون ۳: سکته‌های نیمکره راست
    if 'right' in loc:
        if 'frontal' in loc or 'cortex' in loc or 'radiata' in loc:
            smart_channels += ['F4', 'Fz']
        if 'parietal' in loc or 'occipital' in loc:
            smart_channels += ['P4', 'Pz']

    return list(set(smart_channels))

# [کلاس StandardFBCSP]
class StandardFBCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_bands, n_components=4):
        self.n_bands = n_bands
        self.n_components = n_components
        self.csps_ = []

    def fit(self, X, y):
        self.csps_ = []
        for b in range(self.n_bands):
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True, norm_trace=False)
            csp.fit(X[:, b, :, :], y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        X_feat = []
        for b in range(self.n_bands):
            feat_b = self.csps_[b].transform(X[:, b, :, :])
            X_feat.append(feat_b)
        return np.hstack(X_feat)

# [تابع augment_data]
def augment_data(X_train, y_train, method, aug_factor=2, seed=128):
    np.random.seed(seed)
    if method == 'None':
        return X_train, y_train

    X_aug, y_aug = [], []
    classes = np.unique(y_train)
    n_trials, n_bands, n_channels, n_times = X_train.shape

    if 'AVG' in method:
        for c in classes:
            idx = np.where(y_train == c)[0]
            for _ in range(len(idx) * aug_factor):
                i1, i2 = np.random.choice(idx, 2, replace=True)
                X_aug.append((X_train[i1] + X_train[i2]) / 2.0)
                y_aug.append(c)

    if 'RT' in method:
        n_slices = 3
        edges = np.linspace(0, n_times, n_slices + 1, dtype=int)
        for c in classes:
            idx = np.where(y_train == c)[0]
            for _ in range(len(idx) * aug_factor):
                new_trial = np.zeros((n_bands, n_channels, n_times))
                for s in range(n_slices):
                    t_idx = np.random.choice(idx)
                    new_trial[:, :, edges[s]:edges[s+1]] = X_train[t_idx, :, :, edges[s]:edges[s+1]]
                X_aug.append(new_trial)
                y_aug.append(c)

    if 'RF' in method:
        for c in classes:
            idx = np.where(y_train == c)[0]
            for _ in range(len(idx) * aug_factor):
                new_trial = np.zeros((n_bands, n_channels, n_times))
                for b in range(n_bands):
                    t_idx = np.random.choice(idx)
                    new_trial[b, :, :] = X_train[t_idx, b, :, :]
                X_aug.append(new_trial)
                y_aug.append(c)

    if 'NS' in method:
        std_dev = np.std(X_train, axis=(0, -1), keepdims=True)[0]
        for c in classes:
            idx = np.where(y_train == c)[0]
            for _ in range(aug_factor):
                for i in idx:
                    noise = np.random.normal(0, 0.1 * std_dev, size=X_train[i].shape)
                    X_aug.append(X_train[i] + noise)
                    y_aug.append(c)

    if 'CPS' in method:
        crop_len = int(0.85 * n_times)
        for c in classes:
            idx = np.where(y_train == c)[0]
            for _ in range(aug_factor):
                for i in idx:
                    start = np.random.randint(0, n_times - crop_len + 1)
                    new_trial = np.zeros_like(X_train[i])
                    new_trial[:, :, start:start+crop_len] = X_train[i, :, :, start:start+crop_len]
                    X_aug.append(new_trial)
                    y_aug.append(c)

    if len(X_aug) > 0:
        X_train = np.concatenate([X_train, np.array(X_aug)], axis=0)
        y_train = np.concatenate([y_train, np.array(y_aug)], axis=0)

    return X_train, y_train

def set_seeds(seed=128):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def main():
    set_seeds(128)

    # 1. بارگذاری داینامیک متادیتای سکته بیماران
    participants_path = "data/participants.csv"
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"❌ فایل یافت نشد! مطمئن شوید '{participants_path}' در مسیر اجرای کد است.")

    df_part = pd.read_csv(participants_path)

    # پیدا کردن ستون مربوط به محل ضایعه (مقاوم در برابر تغییر نام ستون)
    loc_col = None
    for col in df_part.columns:
        if any(k in col.lower() for k in ['location', 'lesion', 'stroke', 'description']):
            loc_col = col
            break
    if loc_col is None:
        loc_col = df_part.columns[1]

    # پیدا کردن ستون شناسه سابجکت
    id_col = df_part.columns[0]
    for col in df_part.columns:
        if 'id' in col.lower() or 'sub' in col.lower():
            id_col = col
            break

    BANDS = [(2.0, 6.0), (6.0, 10.0), (10.0, 14.0), (14.0, 18.0), (18.0, 24.0), (24.0, 30.0)]
    TMIN, TMAX = 0.5, 3.5

    subjects = list(range(1, 51))

    # 🎯 تعریف متدها: دیتای خام در برابر روش تطبیقی جدید مبتنی بر ضایعه بالینی
    methods = ['None', 'Adaptive_Aug']

    print(f"🚀 Running Lesion-Adaptive Channel & Augmentation Benchmark on {len(subjects)} subjects...")
    print(f"📊 Using metadata column: '{loc_col}' for anatomical routing and adaptive logic.\n")

    master_results = {m: [] for m in methods}
    master_results['Subject'] = []

    for sub_num in subjects:
        subject_id = f"{sub_num:02d}"
        sub_str_look = f"sub-{subject_id}"

        # استخراج هوشمند و پویا محل سکته برای سابجکت فعلی
        sub_row = df_part[df_part[id_col].astype(str).str.contains(sub_str_look, case=False, na=False)]
        if not sub_row.empty:
            stroke_loc = sub_row[loc_col].values[0]
        else:
            stroke_loc = df_part.iloc[sub_num - 1][loc_col] if sub_num - 1 < len(df_part) else "Unknown"

        chosen_channels = get_smart_channels(stroke_loc)

        print(f"⚡ {sub_str_look} | Lesion: {stroke_loc}")
        print(f"   ↳ 🎯 Channels ({len(chosen_channels)}): {chosen_channels}")

        sub_dict = {m: np.nan for m in methods}

        try:
            edf_file = f"/media/amirmahdi/NewVolume/edffile/sub-{subject_id}/eeg/sub-{subject_id}_task-motor-imagery_eeg.edf"
            if not os.path.exists(edf_file):
                print(f"[WARNING] File not found for sub-{subject_id}")
                master_results['Subject'].append(sub_str_look)
                for m in methods: master_results[m].append(np.nan)
                continue

            raw_base = mne.io.read_raw_edf(edf_file, preload=True, eog=['HEOL', 'HEOR'], verbose=False)
            raw_base.rename_channels({'HEOL' : 'EOG1', 'HEOR' : 'EOG2'}, verbose=False)
            raw_base.set_channel_types({'EOG1' : 'eog', 'EOG2' : 'eog'}, verbose=False)

            # فیلتر کانال‌های انتخاب شده بر اساس آناتومی
            raw_base.pick(chosen_channels, verbose=False)

            try:
                raw_base.set_montage('standard_1005', match_case=False, verbose=False)
                raw_spatial = mne.preprocessing.compute_current_source_density(raw_base, verbose=False)
            except Exception:
                raw_spatial = raw_base.copy().set_eeg_reference(ref_channels='average', verbose=False)

            raw_spatial.resample(500, verbose=False)

            events_df = pd.read_csv("data/task-motor-imagery_events.csv")
            motor_imagery_df = events_df[events_df['value'] == 2]
            onset = motor_imagery_df['onset'] / 1000.0
            duration = motor_imagery_df['duration'] / 1000.0
            description = ['left_hand' if r['trial_type'] == 1 else 'right_hand' for _, r in motor_imagery_df.iterrows()]
            annotations = mne.Annotations(onset=onset, duration=duration, description=description)

            fb_epochs = []
            y_labels = None

            for (l_freq, h_freq) in BANDS:
                raw_filt = raw_spatial.copy().filter(l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
                raw_filt.set_annotations(annotations, verbose=False)
                events, event_id = mne.events_from_annotations(raw_filt, event_id={'left_hand': 1, 'right_hand': 2}, verbose=False)

                epochs = mne.Epochs(raw_filt, events, event_id=event_id, tmin=-0.5, tmax=4.0, baseline=(-0.5, 0), preload=True, verbose=False)
                time_mask = (epochs.times >= TMIN) & (epochs.times <= TMAX)
                fb_epochs.append(epochs.get_data(copy=False)[:, :, time_mask])

                if y_labels is None:
                    y_labels = epochs.events[:, -1]

            X_fb = np.stack(fb_epochs, axis=1)
            y = y_labels

            cv = StratifiedShuffleSplit(n_splits=10, train_size=0.6, random_state=128)

            for method in methods:
                try:
                    fold_acc = []

                    # 🧠 پیاده‌سازی منطق تطبیقی: سوئیچ هوشمند متد بر اساس پاتولوژی ضایعه
                    if method == 'Adaptive_Aug':
                        loc = stroke_loc.lower() if isinstance(stroke_loc, str) else ""
                        if 'pons' in loc or 'medulla' in loc or 'cerebell' in loc:
                            actual_method = 'None'    # مهار آرتیفکت عضلانی ساقه مغز با حذف آگمنتیشن مخرب
                        else:
                            actual_method = 'RF+NS'   # تزریق کاتالیزور فرکانسی+نویز برای شتاب‌دهی به نواحی ساب‌کورتیکال
                    else:
                        actual_method = method

                    for train_idx, test_idx in cv.split(X_fb, y):
                        X_tr, y_tr = X_fb[train_idx], y[train_idx]
                        X_te, y_te = X_fb[test_idx], y[test_idx]

                        X_tr_aug, y_tr_aug = augment_data(X_tr, y_tr, method=actual_method, aug_factor=2, seed=128)

                        current_n_channels = X_tr_aug.shape[2]
                        n_comp = min(4, current_n_channels)

                        pipeline = Pipeline([
                            ('fbcsp', StandardFBCSP(n_bands=len(BANDS), n_components=n_comp)),
                            ('scaler', StandardScaler()),
                            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
                        ])

                        pipeline.fit(X_tr_aug, y_tr_aug)
                        preds = pipeline.predict(X_te)
                        fold_acc.append(accuracy_score(y_te, preds))

                    mean_acc = np.mean(fold_acc)
                    sub_dict[method] = mean_acc
                    print(f" ➜ Strategy: {method:<15} (Action: {actual_method:<5}) | Mean Acc = {mean_acc:.2%}")

                except Exception as method_e:
                    print(f"[ERROR] Strategy {method} failed on sub-{subject_id}: {method_e}")
                    sub_dict[method] = np.nan

            master_results['Subject'].append(sub_str_look)
            for m in methods:
                master_results[m].append(sub_dict[m])

        except Exception as e:
            print(f"[ERROR] General Engine fault on sub-{subject_id}: {e}")
            master_results['Subject'].append(sub_str_look)
            for m in methods:
                master_results[m].append(np.nan)
            continue

    # ==========================================================
    # نمایش خروجی نهایی جدول مقایسه
    # ==========================================================
    print("\n==========================================================================")
    print("📊 FINAL COMPARISON TABLE (Lesion-Adaptive Framework)")
    print("==========================================================================")

    df_results = pd.DataFrame(master_results)

    mean_row = {m: df_results[m].mean(skipna=True) for m in methods}
    mean_row['Subject'] = 'GLOBAL MEAN'
    df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)

    for m in methods:
        df_results[m] = df_results[m].apply(lambda x: f"{x:.2%}" if not np.isnan(x) else "NaN")

    print(df_results.to_string(index=False))
    print("==========================================================================")

    df_results.to_csv("smart_anatomical_fbcsp_lda_results.csv", index=False)
    print("✅ Results saved to 'smart_anatomical_fbcsp_lda_results.csv'")

if __name__ == "__main__":
    main()