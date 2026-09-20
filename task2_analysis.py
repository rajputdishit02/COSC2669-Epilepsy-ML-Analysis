from pathlib import Path
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import mne

from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import MetricFrame

RANDOM_STATE = 42
OUT = Path("task2_outputs")
OUT.mkdir(exist_ok=True)

# -----------------------------
# BEED: dataset, target and class distribution
# -----------------------------
beed_ds = fetch_ucirepo(id=1134)
X_beed = beed_ds.data.features.copy()
y_beed = beed_ds.data.targets.squeeze().astype(int)

class_names = {
    0: "Healthy",
    1: "Generalised seizure",
    2: "Focal seizure",
    3: "Seizure event",
}
class_dist = (
    pd.Series(y_beed)
    .value_counts()
    .sort_index()
    .rename_axis("target")
    .reset_index(name="count")
)
class_dist["class_name"] = class_dist["target"].map(class_names)
class_dist["proportion"] = class_dist["count"] / len(y_beed)
class_dist.to_csv(OUT / "beed_class_distribution.csv", index=False)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

gb_beed = GradientBoostingClassifier(random_state=RANDOM_STATE)
mlp_beed = Pipeline([
    ("scale", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE
    ))
])

# Learning curves: macro-F1 so every class contributes equally.
train_fracs = np.array([0.10, 0.25, 0.40, 0.60, 0.80, 1.00])
lc_rows = []
for model_name, model in [("Gradient Boosting", gb_beed), ("MLP", mlp_beed)]:
    sizes, train_scores, val_scores = learning_curve(
        model,
        X_beed,
        y_beed,
        cv=cv,
        train_sizes=train_fracs,
        scoring="f1_macro",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
        error_score="raise"
    )
    for i, size in enumerate(sizes):
        lc_rows.append({
            "model": model_name,
            "train_size": int(size),
            "train_fraction": float(train_fracs[i]),
            "train_f1_mean": float(train_scores[i].mean()),
            "train_f1_std": float(train_scores[i].std(ddof=1)),
            "validation_f1_mean": float(val_scores[i].mean()),
            "validation_f1_std": float(val_scores[i].std(ddof=1)),
        })

lc_df = pd.DataFrame(lc_rows)
lc_df.to_csv(OUT / "beed_learning_curve.csv", index=False)

for model_name in lc_df["model"].unique():
    d = lc_df[lc_df["model"] == model_name]
    plt.figure(figsize=(7, 5))
    plt.plot(d["train_size"], d["train_f1_mean"], marker="o", label="Training macro-F1")
    plt.plot(d["train_size"], d["validation_f1_mean"], marker="o", label="Validation macro-F1")
    plt.fill_between(
        d["train_size"],
        d["validation_f1_mean"] - d["validation_f1_std"],
        d["validation_f1_mean"] + d["validation_f1_std"],
        alpha=0.2
    )
    plt.xlabel("Training observations")
    plt.ylabel("Macro-F1")
    plt.ylim(0.0, 1.02)
    plt.title(f"BEED learning curve - {model_name}")
    plt.legend()
    plt.tight_layout()
    safe = model_name.lower().replace(" ", "_")
    plt.savefig(OUT / f"beed_learning_curve_{safe}.png", dpi=220)
    plt.close()

# Reproduce out-of-fold predictions and use Fairlearn MetricFrame to inspect
# performance disparity across clinical target classes. These are NOT protected
# demographic groups; the analysis is explicitly a diagnostic of class-specific
# model performance because BEED exposes no row-level demographic attribute.
def beed_oof(model):
    pred = cross_val_predict(model, X_beed, y_beed, cv=cv, method="predict", n_jobs=-1)
    prob = cross_val_predict(model, X_beed, y_beed, cv=cv, method="predict_proba", n_jobs=-1)
    return pred, prob

fair_rows = []
overall_rows = []
for model_name, model in [("Gradient Boosting", gb_beed), ("MLP", mlp_beed)]:
    pred, prob = beed_oof(model)
    sensitive = pd.Series(y_beed).map(class_names)
    mf = MetricFrame(
        metrics={"correct_classification_rate": accuracy_score},
        y_true=y_beed,
        y_pred=pred,
        sensitive_features=sensitive
    )
    by_group = mf.by_group.reset_index()
    by_group.columns = ["clinical_class", "correct_classification_rate"]
    by_group["model"] = model_name
    for _, row in by_group.iterrows():
        fair_rows.append(row.to_dict())

    overall_rows.append({
        "dataset": "BEED",
        "model": model_name,
        "accuracy": accuracy_score(y_beed, pred),
        "precision_macro": precision_score(y_beed, pred, average="macro"),
        "recall_macro": recall_score(y_beed, pred, average="macro"),
        "f1_macro": f1_score(y_beed, pred, average="macro"),
        "roc_auc_macro_ovr": roc_auc_score(y_beed, prob, multi_class="ovr", average="macro"),
        "class_performance_gap_max_minus_min": float(mf.difference(method="between_groups"))
    })

pd.DataFrame(fair_rows).to_csv(OUT / "beed_fairlearn_by_class.csv", index=False)

# -----------------------------
# CHB-MIT: reproduce original preprocessing and LOGO evaluation
# -----------------------------
CHB = Path("task2_chb_raw")
CHB.mkdir(exist_ok=True)
SEIZURE_INTERVALS = {
    "chb01_03.edf": (2996, 3036),
    "chb01_04.edf": (1467, 1494),
    "chb01_15.edf": (1732, 1772),
    "chb01_16.edf": (1015, 1066),
    "chb01_18.edf": (1720, 1810),
    "chb01_21.edf": (327, 420),
    "chb01_26.edf": (1862, 1963),
}
base = "https://physionet.org/files/chbmit/1.0.0/chb01/"
for filename in SEIZURE_INTERVALS:
    path = CHB / filename
    if not path.exists():
        print("Downloading", filename, flush=True)
        urllib.request.urlretrieve(base + filename, path)

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}
FEATURE_NAMES = []
for name in ["mean", "std", "rms", "ptp", "line_length", "zero_cross_rate"]:
    FEATURE_NAMES.extend([f"{name}_mean_ch", f"{name}_std_ch"])
for name in list(BANDS) + ["total_power", "spectral_entropy"]:
    FEATURE_NAMES.extend([f"{name}_mean_ch", f"{name}_std_ch"])

def aggregate_window_features(data, sfreq):
    eps = 1e-12
    ch_mean = np.mean(data, axis=1)
    ch_std = np.std(data, axis=1)
    ch_rms = np.sqrt(np.mean(data ** 2, axis=1))
    ch_ptp = np.ptp(data, axis=1)
    ch_line_length = np.mean(np.abs(np.diff(data, axis=1)), axis=1)
    centered = data - ch_mean[:, None]
    ch_zero_cross = np.mean((centered[:, :-1] * centered[:, 1:]) < 0, axis=1)

    freqs, psd = welch(data, fs=sfreq, nperseg=min(512, data.shape[1]), axis=1)
    total_mask = (freqs >= 0.5) & (freqs <= 45)
    total_power = np.trapezoid(psd[:, total_mask], freqs[total_mask], axis=1) + eps

    relative_band_power = {}
    for band, (low, high) in BANDS.items():
        band_mask = (freqs >= low) & (freqs < high)
        band_power = np.trapezoid(psd[:, band_mask], freqs[band_mask], axis=1)
        relative_band_power[band] = band_power / total_power

    p = psd[:, total_mask]
    p_norm = p / (p.sum(axis=1, keepdims=True) + eps)
    spectral_entropy = -np.sum(p_norm * np.log(p_norm + eps), axis=1) / np.log(p_norm.shape[1])

    channel_features = [
        ch_mean, ch_std, ch_rms, ch_ptp, ch_line_length, ch_zero_cross,
        *[relative_band_power[b] for b in BANDS],
        total_power, spectral_entropy,
    ]
    output = []
    for values in channel_features:
        output.extend([float(np.mean(values)), float(np.std(values))])
    return output

def extract_windows(filepath, seizure_start, seizure_end, window_seconds=5, safety_margin=60):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    n_seizure = int((seizure_end - seizure_start) // window_seconds)
    seizure_starts = [seizure_start + i * window_seconds for i in range(n_seizure)]

    candidate_nonseizure = []
    for start in np.arange(0, duration - window_seconds + 1e-9, window_seconds):
        end = start + window_seconds
        if end <= seizure_start - safety_margin or start >= seizure_end + safety_margin:
            candidate_nonseizure.append(float(start))
    selected_indices = np.linspace(0, len(candidate_nonseizure) - 1, n_seizure, dtype=int)
    nonseizure_starts = [candidate_nonseizure[i] for i in selected_indices]

    rows = []
    for label, starts in [(1, seizure_starts), (0, nonseizure_starts)]:
        for start in starts:
            a = int(round(start * sfreq))
            b = int(round((start + window_seconds) * sfreq))
            data = raw.get_data(start=a, stop=b)
            rows.append(aggregate_window_features(data, sfreq) + [label, Path(filepath).name, start])
    return rows

chb_rows = []
for filename, (start, end) in SEIZURE_INTERVALS.items():
    chb_rows.extend(extract_windows(CHB / filename, start, end))

chb = pd.DataFrame(chb_rows, columns=FEATURE_NAMES + ["label", "recording", "start_sec"])
chb.groupby(["recording", "label"]).size().rename("count").reset_index().to_csv(
    OUT / "chb_class_distribution_by_recording.csv", index=False
)

X_chb = chb[FEATURE_NAMES].to_numpy()
y_chb = chb["label"].to_numpy()
groups = chb["recording"].to_numpy()
logo = LeaveOneGroupOut()

gb_chb = GradientBoostingClassifier(random_state=RANDOM_STATE)
mlp_chb = Pipeline([
    ("scale", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        early_stopping=False,
        random_state=RANDOM_STATE
    ))
])

def grouped_predictions(model):
    pred = np.zeros_like(y_chb)
    prob = np.zeros(len(y_chb), dtype=float)
    for train_idx, test_idx in logo.split(X_chb, y_chb, groups):
        fit = clone(model)
        fit.fit(X_chb[train_idx], y_chb[train_idx])
        pred[test_idx] = fit.predict(X_chb[test_idx])
        prob[test_idx] = fit.predict_proba(X_chb[test_idx])[:, 1]
    return pred, prob

def safe_precision(y, p):
    return precision_score(y, p, zero_division=0)
def safe_recall(y, p):
    return recall_score(y, p, zero_division=0)
def safe_f1(y, p):
    return f1_score(y, p, zero_division=0)

recording_rows = []
for model_name, model in [("Gradient Boosting", gb_chb), ("MLP", mlp_chb)]:
    pred, prob = grouped_predictions(model)
    overall_rows.append({
        "dataset": "CHB-MIT",
        "model": model_name,
        "accuracy": accuracy_score(y_chb, pred),
        "precision_macro": precision_score(y_chb, pred, average="binary", zero_division=0),
        "recall_macro": recall_score(y_chb, pred, average="binary", zero_division=0),
        "f1_macro": f1_score(y_chb, pred, average="binary", zero_division=0),
        "roc_auc_macro_ovr": roc_auc_score(y_chb, prob),
        "class_performance_gap_max_minus_min": np.nan
    })
    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": safe_precision,
            "recall": safe_recall,
            "f1": safe_f1,
        },
        y_true=y_chb,
        y_pred=pred,
        sensitive_features=groups
    )
    bg = mf.by_group.reset_index().rename(columns={"sensitive_feature_0": "recording"})
    if "recording" not in bg.columns:
        bg = bg.rename(columns={bg.columns[0]: "recording"})
    bg["model"] = model_name
    for _, row in bg.iterrows():
        recording_rows.append(row.to_dict())

pd.DataFrame(recording_rows).to_csv(OUT / "chb_fairlearn_by_recording.csv", index=False)
pd.DataFrame(overall_rows).to_csv(OUT / "task2_model_metrics.csv", index=False)

summary = []
summary.append(f"BEED shape: {X_beed.shape[0]} rows x {X_beed.shape[1]} features")
summary.append("BEED target: y; four balanced classes of 2,000 each")
summary.append(f"CHB-MIT modelling table: {chb.shape[0]} windows; seizure={int((y_chb==1).sum())}; non-seizure={int((y_chb==0).sum())}")
summary.append(f"CHB-MIT recordings used: {len(np.unique(groups))}; all from patient chb01")
summary.append("Important correction to Task 1 report: chb01_01.edf was checked in the original notebook but was not used to generate modelling rows.")
(OUT / "methodology_summary.txt").write_text("\n".join(summary), encoding="utf-8")

print(pd.DataFrame(overall_rows))
print(lc_df)
print(pd.DataFrame(recording_rows))
