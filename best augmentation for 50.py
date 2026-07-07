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

# 🎯 ابزارهای هندسه ریمانی و انتخاب ویژگی مبتنی بر اطلاعات متقابل
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.feature_selection import SelectKBest, mutual_info_classif

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==========================================================
# تابع استراتژی هوشمند انتخاب کانال بر اساس آناتومی و سمت فلج بدن
# ==========================================================
def get_smart_channels(location_str, paralysis_side):
    base_channels = ['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4']
    if pd.isna(location_str) or not isinstance(location_str, str):
        return base_channels

    loc = location_str.lower()
    p_side = str(paralysis_side).lower() if not pd.isna(paralysis_side) else ""

    if 'pons' in loc or 'medulla' in loc or 'cerebell' in loc:
        return ['FCz', 'C3', 'Cz', 'C4', 'CPz']

    smart_channels = base_channels.copy()

    if 'left' in loc or 'right' in p_side:
        if any(k in loc for k in ['frontal', 'cortex', 'radiata', 'capsule', 'parietal', 'semiovale']):
            smart_channels += ['F3', 'Fz', 'P3', 'Pz']

    if 'right' in loc or 'left' in p_side:
        if any(k in loc for k in ['frontal', 'cortex', 'radiata', 'capsule', 'parietal', 'occipital', 'semiovale']):
            smart_channels += ['F4', 'Fz', 'P4', 'Pz']

    return list(set(smart_channels))


# ==========================================================
# کلاس پیاده‌سازی الگوریتم پیشنهادی مقاله (FBCSP-ASP مبتنی بر PSO)
# ==========================================================
class ImprovedFBCSP_ASP(BaseEstimator, TransformerMixin):
    def __init__(self, n_bands, n_components=4, n_asp_components=2, n_particles=10, max_iter=10, k_features=24):
        self.n_bands = n_bands
        self.n_components = n_components
        self.n_asp_components = n_asp_components  # تعداد فیلترهای تطبیقی در هر باند فرکانسی
        self.n_particles = n_particles            # تعداد ذرات در الگوریتم PSO
        self.max_iter = max_iter                  # تعداد تکرارهای PSO
        self.k_features = k_features              # تعداد ویژگی‌های نهایی انتخاب‌شده پس از مرحله هرس اطلاعات متقابل
        self.csps_ = []
        self.asp_filters_ = []
        self.selector_ = None

    def _optimize_asp_filter(self, covs, y, seed):
        """ بهینه‌سازی فیلترهای مکانی تطبیقی مبتنی بر بیشینه‌سازی تفاوت انرژی بین کلاس‌ها با PSO """
        rng = np.random.default_rng(seed)
        n_trials, n_channels, _ = covs.shape
        classes = np.unique(y)

        if len(classes) != 2:
            return rng.standard_normal((n_channels,))

        idx1 = (y == classes[0])
        idx2 = (y == classes[1])

        # مقداردهی اولیه موقعیت ذرات روی کره واحد فضایی
        X_p = rng.standard_normal((self.n_particles, n_channels))
        X_p /= (np.linalg.norm(X_p, axis=1, keepdims=True) + 1e-8)

        V_p = rng.standard_normal((self.n_particles, n_channels)) * 0.1

        pbest = np.copy(X_p)
        pbest_fit = np.full(self.n_particles, -1e9)

        gbest = np.copy(X_p[0])
        gbest_fit = -1e9

        # حلقه بهینه‌سازی ذرات
        for it in range(self.max_iter):
            for i in range(self.n_particles):
                w = X_p[i]
                # محاسبات تمام‌برداری سریع ماتریس انرژی سیگنال فیلتر شده برای تک‌تک تریال‌ها
                tmp = covs @ w
                E = np.sum(tmp * w, axis=1)
                E = np.log(np.maximum(E, 1e-8)) # تبدیل لگاریتمی مرسوم ویژگی‌های انرژی

                m1, m2 = np.mean(E[idx1]), np.mean(E[idx2])
                v1 = np.var(E[idx1]) + 1e-6
                v2 = np.var(E[idx2]) + 1e-6

                # تابع برازش (Fitness Function): بیشینه‌سازی فاصله بین‌کلاسی و کمینه‌سازی واریانس درون‌کلاسی
                fit_val = ((m1 - m2) ** 2) / (v1 + v2)

                if fit_val > pbest_fit[i]:
                    pbest_fit[i] = fit_val
                    pbest[i] = np.copy(w)

                if fit_val > gbest_fit:
                    gbest_fit = fit_val
                    gbest = np.copy(w)

            # به‌روزرسانی سرعت و موقعیت توده ذرات
            w_inertia = 0.729
            c1, c2 = 1.49445, 1.49445
            r1 = rng.uniform(0, 1, (self.n_particles, n_channels))
            r2 = rng.uniform(0, 1, (self.n_particles, n_channels))

            V_p = w_inertia * V_p + c1 * r1 * (pbest - X_p) + c2 * r2 * (gbest - X_p)
            X_p = X_p + V_p
            X_p /= (np.linalg.norm(X_p, axis=1, keepdims=True) + 1e-8) # نرمال‌سازی مجدد فیلتر

        return gbest

    def fit(self, X, y):
        self.csps_ = []
        self.asp_filters_ = []
        all_train_features = []

        for b in range(self.n_bands):
            X_band = X[:, b, :, :]

            # ۱. استخراج ویژگی‌های فیلتر مکانی مشترک سنتی (CSP)
            csp = CSP(n_components=self.n_components, reg='ledoit_wolf', log=True, norm_trace=False)
            csp.fit(X_band, y)
            self.csps_.append(csp)
            feat_csp = csp.transform(X_band)

            # ۲. استخراج ویژگی‌های فیلتر مکانی تطبیقی (ASP) به کمک PSO
            covs = np.matmul(X_band, X_band.transpose(0, 2, 1)) / X_band.shape[-1]
            band_filters = []
            feat_asp_list = []

            for c_idx in range(self.n_asp_components):
                # اجرای PSO با دانه‌های تصادفی مختلف جهت دریافت فیلترهای مکمل و متنوع
                w_best = self._optimize_asp_filter(covs, y, seed=128 + b * 10 + c_idx)
                band_filters.append(w_best)

                tmp = covs @ w_best
                E = np.sum(tmp * w_best, axis=1)
                feat_asp_w = np.log(np.maximum(E, 1e-8)).reshape(-1, 1)
                feat_asp_list.append(feat_asp_w)

            self.asp_filters_.append(band_filters)

            # ادغام و تلفیق ویژگی‌های هر دو الگو در باند جاری
            feat_band = np.hstack([feat_csp] + feat_asp_list)
            all_train_features.append(feat_band)

        X_all_feat = np.hstack(all_train_features)

        # ۳. مرحله دوم: انتخاب برترین ویژگی‌ها بر اساس Mutual Information برای حذف نویز و ابعاد زاید
        actual_k = min(self.k_features, X_all_feat.shape[1])
        self.selector_ = SelectKBest(score_func=mutual_info_classif, k=actual_k)
        self.selector_.fit(X_all_feat, y)

        return self

    def transform(self, X):
        all_features = []
        for b in range(self.n_bands):
            X_band = X[:, b, :, :]

            feat_csp = self.csps_[b].transform(X_band)

            covs = np.matmul(X_band, X_band.transpose(0, 2, 1)) / X_band.shape[-1]
            feat_asp_list = []

            for w_best in self.asp_filters_[b]:
                tmp = covs @ w_best
                E = np.sum(tmp * w_best, axis=1)
                feat_asp_w = np.log(np.maximum(E, 1e-8)).reshape(-1, 1)
                feat_asp_list.append(feat_asp_w)

            feat_band = np.hstack([feat_csp] + feat_asp_list)
            all_features.append(feat_band)

        X_all_feat = np.hstack(all_features)
        return self.selector_.transform(X_all_feat)


# ==========================================================
# کلاس محاسبات هندسه ریمانی در ساختار فیلتربانک (FBRTS)
# ==========================================================
class StandardFBRiemann(BaseEstimator, TransformerMixin):
    def __init__(self, n_bands):
        self.n_bands = n_bands
        self.covs_ = []
        self.ts_ = []

    def fit(self, X, y):
        self.covs_ = []
        self.ts_ = []
        for b in range(self.n_bands):
            cov = Covariances(estimator='lwf')
            ts = TangentSpace(metric='riemann')
            cov_matrices = cov.fit_transform(X[:, b, :, :])
            ts.fit(cov_matrices, y)
            self.covs_.append(cov)
            self.ts_.append(ts)
        return self

    def transform(self, X):
        X_feat = []
        for b in range(self.n_bands):
            cov_matrices = self.covs_[b].transform(X[:, b, :, :])
            feat_b = self.ts_[b].transform(cov_matrices)
            X_feat.append(feat_b)
        return np.hstack(X_feat)


# [تابع augment_data بدون تغییر]
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

    participants_path = "data/participants.csv"
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"❌ فایل یافت نشد! مطمئن شوید '{participants_path}' در مسیر اجرای کد است.")

    df_part = pd.read_csv(participants_path)

    id_col = next((c for c in df_part.columns if 'id' in c.lower() or 'sub' in c.lower()), df_part.columns[0])
    loc_col = next((c for c in df_part.columns if any(k in c.lower() for k in ['location', 'lesion', 'stroke', 'description'])), df_part.columns[7])
    para_col = next((c for c in df_part.columns if 'paralysis' in c.lower() or 'side' in c.lower()), None)
    nihss_col = next((c for c in df_part.columns if 'nihss' in c.lower()), None)
    mbi_col = next((c for c in df_part.columns if 'mbi' in c.lower() or 'barthel' in c.lower()), None)

    BANDS = [(2.0, 6.0), (6.0, 10.0), (10.0, 14.0), (14.0, 18.0), (18.0, 24.0), (24.0, 30.0)]
    TMIN, TMAX = 0.5, 3.0

    subjects = list(range(1, 51))
    methods = ['None', 'Adaptive_Riemann_Ensemble']

    print(f"🚀 Running Hybrid Clinical-Adaptive + Asymmetric Riemannian Ensemble Framework...")
    print(f"📊 FBCSP-ASP branch accepts Augmented data | Riemannian branch locked to Pristine data.\n")

    master_results = {m: [] for m in methods}
    master_results['Subject'] = []

    for sub_num in subjects:
        subject_id = f"{sub_num:02d}"
        sub_str_look = f"sub-{subject_id}"

        sub_row = df_part[df_part[id_col].astype(str).str.contains(sub_str_look, case=False, na=False)]

        if not sub_row.empty:
            stroke_loc = sub_row[loc_col].values[0]
            paralysis_side = sub_row[para_col].values[0] if para_col else "Unknown"

            try: nihss_val = float(sub_row[nihss_col].values[0]) if nihss_col and not pd.isna(sub_row[nihss_col].values[0]) else 3.0
            except: nihss_val = 3.0

            try: mbi_val = float(sub_row[mbi_col].values[0]) if mbi_col and not pd.isna(sub_row[mbi_col].values[0]) else 75.0
            except: mbi_val = 75.0
        else:
            stroke_loc, paralysis_side, nihss_val, mbi_val = "Unknown", "Unknown", 3.0, 75.0

        chosen_channels = get_smart_channels(stroke_loc, paralysis_side)

        print(f"⚡ {sub_str_look} | Location: {stroke_loc:<40} | Side: {paralysis_side:<5}")
        print(f"   ↳ 🎯 Routed Channels ({len(chosen_channels)}): {chosen_channels}")

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
                raw_filt = raw_spatial.copy().filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_design='firwin2', verbose=False)
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

                    if method == 'Adaptive_Riemann_Ensemble':
                        loc = stroke_loc.lower() if isinstance(stroke_loc, str) else ""
                        if 'pons' in loc or 'medulla' in loc or 'cerebell' in loc:
                            actual_method = 'None'
                            dyn_aug_factor = 2
                        elif any(k in loc for k in ['cortex', 'frontal', 'parietal', 'radiata', 'capsule', 'semiovale']):
                            actual_method = 'AVG+NS'
                            dyn_aug_factor = 1 if (nihss_val >= 6 or mbi_val < 60) else 1
                        else:
                            actual_method = 'AVG+NS'
                            dyn_aug_factor = 1 if (nihss_val >= 6 or mbi_val < 60) else 1
                    else:
                        actual_method = 'None'
                        dyn_aug_factor = 2

                    for train_idx, test_idx in cv.split(X_fb, y):
                        X_tr, y_tr = X_fb[train_idx], y[train_idx]
                        X_te, y_te = X_fb[test_idx], y[test_idx]

                        # تولید دیتای افزوده شده فقط برای بازوی FBCSP-ASP
                        X_tr_aug, y_tr_aug = augment_data(X_tr, y_tr, method=actual_method, aug_factor=dyn_aug_factor, seed=128)

                        current_n_channels = X_tr_aug.shape[2]
                        n_comp = min(4, current_n_channels)

                        # 🛠️ بازنشانی پایپلاین اول: استفاده از الگوریتم برتر و تطبیقی مقاله (FBCSP_ASP) به همراه هرس ویژگی
                        pipe_fbcsp = Pipeline([
                            ('fbcsp', ImprovedFBCSP_ASP(n_bands=len(BANDS), n_components=n_comp, n_asp_components=2, n_particles=30, max_iter=20, k_features=16)),
                            ('scaler', StandardScaler()),
                            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
                        ])

                        # ساخت پایپلاین دوم: هندسه ریمانی
                        pipe_riemann = Pipeline([
                            ('fbriemann', StandardFBRiemann(n_bands=len(BANDS))),
                            ('scaler', StandardScaler()),
                            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
                        ])

                        # فیت کردن مدل‌ها
                        pipe_fbcsp.fit(X_tr_aug, y_tr_aug)  # فیت با دیتای آگمنت‌شده
                        pipe_riemann.fit(X_tr, y_tr)        # فیت دیتای اورجینال جهت حفظ ساختار منیفولد ریمانی

                        # استخراج احتمالات کالیبره شده سافت (Soft-Voting Mechanism)
                        proba_fbcsp = pipe_fbcsp.predict_proba(X_te)
                        proba_riemann = pipe_riemann.predict_proba(X_te)

                        # میانگین‌گیری وزن‌دار احتمالات دو سیستم
                        mean_proba = (proba_fbcsp + proba_riemann) / 2.0

                        classes = pipe_fbcsp.classes_
                        preds = classes[np.argmax(mean_proba, axis=1)]

                        fold_acc.append(accuracy_score(y_te, preds))

                    mean_acc = np.mean(fold_acc)
                    sub_dict[method] = mean_acc
                    print(f" ➜ Strategy: {method:<25} | Mean Acc = {mean_acc:.2%}")

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
    # نمایش خروجی نهایی جدول مقایسه تغییریافته
    # ==========================================================
    print("\n==========================================================================")
    print("📊 FINAL COMPARISON TABLE (Asymmetric Adaptive Riemann Ensemble with FBCSP-ASP)")
    print("==========================================================================")

    df_results = pd.DataFrame(master_results)

    mean_row = {m: df_results[m].mean(skipna=True) for m in methods}
    mean_row['Subject'] = 'GLOBAL MEAN'
    df_results = pd.concat([df_results, pd.DataFrame([mean_row])], ignore_index=True)

    for m in methods:
        df_results[m] = df_results[m].apply(lambda x: f"{x:.2%}" if not np.isnan(x) else "NaN")

    print(df_results.to_string(index=False))
    print("==========================================================================")

    df_results.to_csv("asymmetric_riemann_ensemble_results.csv", index=False)
    print("✅ Results saved to 'asymmetric_riemann_ensemble_results.csv'")

if __name__ == "__main__":
    main()