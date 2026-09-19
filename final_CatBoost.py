
import os
import ntpath
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from catboost import CatBoostClassifier


# =============================================================================
# 0. LOCKED FINAL SETTINGS
# =============================================================================
DATA_FILE = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.19\最终建模\训练集_6.xlsx"
EXTERNAL_DATA_FILE = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.19\最终建模\外部验证集_6.xlsx"

TARGET_COL = "Infection"
POS_LABEL = 1
SEX_COL = "sex"

RANDOM_STATE = 42
THRESHOLD = 0.11

SAMPLING_RATIO = 0.41
K_NEIGHBORS = 11

MODEL_PARAMS = {
    "iterations": 70,
    "learning_rate": 0.13,
    "depth": 1,
    "l2_leaf_reg": 18,
    "random_strength": 11,
    "bagging_temperature": 5,
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
}

BOOTSTRAP_N = 2000
BOOTSTRAP_RANDOM_STATE = 2026

OUTPUT_ROOT = r"D:\萝卜投稿专用\Original Research - Spinal Infection Following Internal Fixation Surgery\插值选择\Xgboost\lr=0.19\最终建模"


# =============================================================================
# 1. PLOT STYLE
# =============================================================================
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

COLOR_TRAIN = "#4E79A7"
COLOR_EXTERNAL = "#59A14F"


# =============================================================================
# 2. HELPERS
# =============================================================================
def resolve_data_file(path):
    if os.path.exists(path):
        return path

    base = ntpath.basename(path)
    for folder in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        candidate = os.path.join(folder, base)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"找不到数据文件：{path}")


def make_output_dir():
    out = os.path.join(
        OUTPUT_ROOT,
        f"CatBoost_PRIMARY_FINAL_{datetime.now():%Y%m%d_%H%M%S}"
    )
    os.makedirs(out, exist_ok=True)
    return out


class ContinuousOnlyScaler(BaseEstimator, TransformerMixin):
    """Z-score continuous predictors; keep sex unchanged as 0/1."""

    def __init__(self, sex_col=SEX_COL):
        self.sex_col = sex_col

    def fit(self, X, y=None):
        X = X.copy()
        self.columns_ = list(X.columns)
        self.continuous_ = [c for c in self.columns_ if c != self.sex_col]

        self.scaler_ = StandardScaler()
        if self.continuous_:
            self.scaler_.fit(X[self.continuous_])
        return self

    def transform(self, X):
        X = X.copy()
        out = X.astype(float).copy()

        if self.continuous_:
            out.loc[:, self.continuous_] = self.scaler_.transform(
                X[self.continuous_]
            )

        if self.sex_col in out.columns:
            out[self.sex_col] = X[self.sex_col].astype(int)

        return out


class SMOTENCSampler(BaseEstimator):
    def __init__(
        self,
        sampling_strategy=1.0,
        random_state=42,
        k_neighbors=5
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_arr = np.asarray(y)

        minority_n = int(pd.Series(y_arr).value_counts().min())
        k = max(1, min(int(self.k_neighbors), minority_n - 1))

        if SEX_COL in X_df.columns:
            sampler = SMOTENC(
                categorical_features=[X_df.columns.get_loc(SEX_COL)],
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=k,
            )
        else:
            sampler = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=k,
            )

        X_res, y_res = sampler.fit_resample(X_df, y_arr)

        return (
            pd.DataFrame(X_res, columns=X_df.columns),
            np.asarray(y_res)
        )


def normalize_sex_name(df):
    matches = [c for c in df.columns if str(c).strip().lower() == "sex"]
    if len(matches) > 1:
        raise ValueError(f"检测到多个Sex列：{matches}")
    if len(matches) == 1 and matches[0] != SEX_COL:
        df = df.rename(columns={matches[0]: SEX_COL})
    return df


def load_training_data(path):
    path = resolve_data_file(path)
    df = normalize_sex_name(pd.read_excel(path).dropna(axis=1, how="all"))

    if TARGET_COL not in df.columns:
        raise ValueError(f"训练集找不到 {TARGET_COL}")

    df = df.dropna(subset=[TARGET_COL]).copy()

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid = y.notna()

    X = df.loc[valid].drop(columns=[TARGET_COL]).reset_index(drop=True)
    y = (y.loc[valid].astype(int).reset_index(drop=True) == POS_LABEL).astype(int)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if SEX_COL in X.columns:
        if X[SEX_COL].isna().any():
            raise ValueError("sex存在缺失。")
        if not set(X[SEX_COL].unique()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError("sex必须为0/1。")

    medians = X.median(numeric_only=True)
    if X.isna().any().any():
        print("训练集存在缺失：用训练集自身中位数填补。")
        X = X.fillna(medians)

    if y.nunique() != 2:
        raise ValueError("训练集结局必须包含0和1。")

    return X, y, medians, path


def load_external_data(path, feature_columns, train_medians):
    path = resolve_data_file(path)
    df = normalize_sex_name(pd.read_excel(path).dropna(axis=1, how="all"))

    if TARGET_COL not in df.columns:
        raise ValueError(f"外部验证集找不到 {TARGET_COL}")

    df = df.dropna(subset=[TARGET_COL]).copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "外部验证集缺少最终模型特征：\n" + "\n".join(missing)
        )

    X = df.loc[:, feature_columns].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if X.isna().any().any():
        print("外部验证集存在缺失：只用训练集同名变量中位数填补。")
        X = X.fillna(train_medians)

    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = (y.loc[valid].astype(int).reset_index(drop=True) == POS_LABEL).astype(int)

    if SEX_COL in X.columns:
        if X[SEX_COL].isna().any():
            raise ValueError("外部sex存在缺失。")
        if not set(X[SEX_COL].unique()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError("外部sex必须为0/1。")

    return X, y, path


def build_final_pipeline():
    return ImbPipeline([
        ("scaler", ContinuousOnlyScaler()),
        (
            "resampler",
            SMOTENCSampler(
                sampling_strategy=SAMPLING_RATIO,
                random_state=RANDOM_STATE,
                k_neighbors=K_NEIGHBORS,
            ),
        ),
        ("model", CatBoostClassifier(**MODEL_PARAMS)),
    ])


def calc_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "AUPRC": average_precision_score(y_true, y_prob),
        "AUROC": roc_auc_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV": precision_score(y_true, pred, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
        "Brier": brier_score_loss(y_true, y_prob),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def bootstrap_external(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    point = calc_metrics(y_true, y_prob)

    metrics = [
        "AUPRC", "AUROC", "Accuracy", "Balanced_Accuracy",
        "Sensitivity", "Specificity", "PPV", "NPV",
        "F1", "MCC", "Brier"
    ]

    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    values = {m: [] for m in metrics}

    valid_b = 0
    while valid_b < BOOTSTRAP_N:
        idx = rng.integers(0, n, n)
        yy = y_true[idx]

        if np.unique(yy).size < 2:
            continue

        mm = calc_metrics(yy, y_prob[idx])
        for metric in metrics:
            values[metric].append(mm[metric])

        valid_b += 1

    rows = []
    for metric in metrics:
        lo, hi = np.percentile(values[metric], [2.5, 97.5])
        rows.append({
            "Dataset": "External_Test",
            "Metric": metric,
            "Point_Estimate": point[metric],
            "CI_Lower": lo,
            "CI_Upper": hi,
            "Bootstrap_N": BOOTSTRAP_N,
            "Threshold": THRESHOLD,
        })

    for metric in ["TP", "FP", "TN", "FN"]:
        rows.append({
            "Dataset": "External_Test",
            "Metric": metric,
            "Point_Estimate": point[metric],
            "CI_Lower": np.nan,
            "CI_Upper": np.nan,
            "Bootstrap_N": np.nan,
            "Threshold": THRESHOLD,
        })

    return pd.DataFrame(rows)



def calibration_intercept_slope(y_true, y_prob):
    """Calibration intercept and slope from y ~ intercept + slope * logit(p)."""
    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))

    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0], dtype=float)

    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        new_beta = np.linalg.pinv(XtW @ X) @ (XtW @ z)
        if np.max(np.abs(new_beta - beta)) < 1e-10:
            beta = new_beta
            break
        beta = new_beta

    return float(beta[0]), float(beta[1])


def make_calibration_summary(y_true, y_prob, dataset_name):
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return pd.DataFrame([{
        "Dataset": dataset_name,
        "N": int(len(y_true)),
        "Observed_Prevalence": float(np.mean(y_true)),
        "Mean_Predicted_Probability": float(np.mean(y_prob)),
        "Brier": brier_score_loss(y_true, y_prob),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
    }])


def make_probability_bins(y_true, y_prob, dataset_name):
    """Fixed probability bands for checking whether displayed risk matches observed risk."""
    df = pd.DataFrame({
        "Observed": np.asarray(y_true).astype(int),
        "Probability": np.asarray(y_prob).astype(float),
    })

    edges = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 1.000001]
    labels = ["0-<10%", "10-<20%", "20-<30%", "30-<40%", "40-<50%", ">=50%"]
    df["Probability_Band"] = pd.cut(
        df["Probability"], bins=edges, labels=labels, right=False, include_lowest=True
    )

    rows = []
    for label in labels:
        g = df[df["Probability_Band"] == label]
        n = len(g)
        rows.append({
            "Dataset": dataset_name,
            "Probability_Band": label,
            "N": n,
            "Positive_N": int(g["Observed"].sum()) if n else 0,
            "Observed_Infection_Rate": float(g["Observed"].mean()) if n else np.nan,
            "Mean_Predicted_Probability": float(g["Probability"].mean()) if n else np.nan,
            "Median_Predicted_Probability": float(g["Probability"].median()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def make_candidate_threshold_table(y_true, y_prob, dataset_name):
    """Describe candidate operating points; this does NOT search for a new best threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    rows = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else np.nan

        rows.append({
            "Dataset": dataset_name,
            "Candidate_Threshold": threshold,
            "Current_Classification_Threshold": "Yes" if np.isclose(threshold, THRESHOLD) else "No",
            "Predicted_High_Risk_N": int(pred.sum()),
            "Predicted_High_Risk_Percent": float(pred.mean()),
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "False_Positive_Rate": 1 - specificity if not np.isnan(specificity) else np.nan,
            "PPV": ppv,
            "NPV": npv,
            "F1": f1,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        })
    return pd.DataFrame(rows)


def make_dca_table(y_true, y_prob, dataset_name):
    """Decision-curve net benefit over clinically plausible thresholds."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    rows = []
    for threshold in np.arange(0.05, 0.51, 0.01):
        pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        odds = threshold / (1 - threshold)
        model_nb = (tp / n) - (fp / n) * odds
        treat_all_nb = prevalence - (1 - prevalence) * odds
        rows.append({
            "Dataset": dataset_name,
            "Threshold": float(round(threshold, 2)),
            "Model_Net_Benefit": float(model_nb),
            "Treat_All_Net_Benefit": float(treat_all_nb),
            "Treat_None_Net_Benefit": 0.0,
        })
    return pd.DataFrame(rows)


def save_added_plots(y_train, p_train, y_ext, p_ext, dca_ext, output_dir):
    # Calibration plot (PNG only)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=600)
    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, color="gray", label="Perfect calibration")

    for y, p, label, color in [
        (y_train, p_train, "Full training", COLOR_TRAIN),
        (y_ext, p_ext, "External test", COLOR_EXTERNAL),
    ]:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8, color=color, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed infection rate")
    ax.set_title("CatBoost Calibration Curve")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "Final_CatBoost_Calibration.png"),
        dpi=600, bbox_inches="tight"
    )
    plt.close(fig)

    # External decision curve (PNG only)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=600)
    ax.plot(
        dca_ext["Threshold"], dca_ext["Model_Net_Benefit"],
        linewidth=2.0, color=COLOR_EXTERNAL, label="CatBoost"
    )
    ax.plot(
        dca_ext["Threshold"], dca_ext["Treat_All_Net_Benefit"],
        "--", linewidth=1.2, color="gray", label="Treat all"
    )
    ax.plot(
        dca_ext["Threshold"], dca_ext["Treat_None_Net_Benefit"],
        ":", linewidth=1.2, color="black", label="Treat none"
    )
    ax.axvline(THRESHOLD, linestyle="--", linewidth=1.0, color="black", alpha=0.7)
    ax.set_xlim(0.05, 0.50)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("External Decision Curve Analysis")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "Final_CatBoost_External_DCA.png"),
        dpi=600, bbox_inches="tight"
    )
    plt.close(fig)


def save_scaler(final_pipeline, X_train, output_dir):
    scaler = final_pipeline.named_steps["scaler"]

    rows = []
    for feature in X_train.columns:
        if feature in scaler.continuous_:
            i = scaler.continuous_.index(feature)
            rows.append({
                "Feature": feature,
                "Training_Mean": float(scaler.scaler_.mean_[i]),
                "Training_SD": float(scaler.scaler_.scale_[i]),
                "Standardized": "Yes",
            })
        else:
            rows.append({
                "Feature": feature,
                "Training_Mean": np.nan,
                "Training_SD": np.nan,
                "Standardized": "No (binary sex)",
            })

    pd.DataFrame(rows).to_excel(
        os.path.join(output_dir, "00_final_training_scaler.xlsx"),
        index=False
    )


def save_plots(y_train, p_train, y_ext, p_ext, output_dir):
    # ROC
    fig, ax = plt.subplots(figsize=(6, 5), dpi=600)

    fpr, tpr, _ = roc_curve(y_train, p_train)
    ax.plot(
        fpr, tpr, linewidth=1.8, color=COLOR_TRAIN,
        label=f"Full training AUC = {roc_auc_score(y_train, p_train):.3f}"
    )

    fpr, tpr, _ = roc_curve(y_ext, p_ext)
    ax.plot(
        fpr, tpr, linewidth=2.0, color=COLOR_EXTERNAL,
        label=f"External test AUC = {roc_auc_score(y_ext, p_ext):.3f}"
    )

    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CatBoost Final ROC Curve")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "Final_CatBoost_ROC.png"),
        dpi=600, bbox_inches="tight"
    )
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6, 5), dpi=600)

    precision, recall, _ = precision_recall_curve(y_train, p_train)
    ax.plot(
        recall, precision, linewidth=1.8, color=COLOR_TRAIN,
        label=f"Full training AUPRC = {average_precision_score(y_train, p_train):.3f}"
    )

    precision, recall, _ = precision_recall_curve(y_ext, p_ext)
    ax.plot(
        recall, precision, linewidth=2.0, color=COLOR_EXTERNAL,
        label=f"External test AUPRC = {average_precision_score(y_ext, p_ext):.3f}"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("CatBoost Final Precision-Recall Curve")
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "Final_CatBoost_PR.png"),
        dpi=600, bbox_inches="tight"
    )
    plt.close(fig)



def make_oof_results(X, y):
    
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_arr = np.asarray(y).astype(int)
    oof_prob = np.full(len(y_arr), np.nan, dtype=float)
    oof_fold = np.full(len(y_arr), -1, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_arr), start=1):
        fold_pipeline = build_final_pipeline()
        fold_pipeline.fit(X.iloc[train_idx].copy(), y_arr[train_idx])
        oof_prob[val_idx] = fold_pipeline.predict_proba(
            X.iloc[val_idx].copy()
        )[:, 1]
        oof_fold[val_idx] = fold

    if np.isnan(oof_prob).any() or (oof_fold < 1).any():
        raise RuntimeError("OOF prediction generation failed: incomplete predictions.")

    oof_predictions = pd.DataFrame({
        "Row_Index": np.arange(len(y_arr)),
        "Fold": oof_fold,
        "Observed": y_arr,
        "OOF_Probability": oof_prob,
        "OOF_Prediction_at_0.11": (oof_prob >= THRESHOLD).astype(int),
    })

    oof_metrics = calc_metrics(y_arr, oof_prob)
    oof_summary = pd.DataFrame([{
        "Dataset": "Training_OOF_5Fold",
        "N": int(len(y_arr)),
        "Positive_N": int(y_arr.sum()),
        "Positive_Rate": float(y_arr.mean()),
        **oof_metrics,
    }])

    oof_calibration = make_calibration_summary(
        y_arr, oof_prob, "Training_OOF_5Fold"
    )
    oof_probability_bands = make_probability_bins(
        y_arr, oof_prob, "Training_OOF_5Fold"
    )
    oof_candidate_thresholds = make_candidate_threshold_table(
        y_arr, oof_prob, "Training_OOF_5Fold"
    )
    oof_dca = make_dca_table(
        y_arr, oof_prob, "Training_OOF_5Fold"
    )

    return (
        oof_summary,
        oof_calibration,
        oof_probability_bands,
        oof_candidate_thresholds,
        oof_dca,
        oof_predictions,
    )


# =============================================================================
# 3. FINAL FIT
# =============================================================================
def main():
    output_dir = make_output_dir()

    X_train, y_train, train_medians, train_file = load_training_data(DATA_FILE)
    X_ext, y_ext, external_file = load_external_data(
        EXTERNAL_DATA_FILE,
        feature_columns=list(X_train.columns),
        train_medians=train_medians,
    )

    print("=" * 80)
    print("PRIMARY FINAL CATBOOST")
    print("NO CV / NO TUNING / NO THRESHOLD SEARCH")
    print(f"Training: {train_file}")
    print(f"External: {external_file}")
    print(f"Train N={len(y_train)}, positive={int(y_train.sum())}")
    print(f"External N={len(y_ext)}, positive={int(y_ext.sum())}")
    print(f"Predictors={X_train.shape[1]}")
    print(f"SMOTENC ratio={SAMPLING_RATIO}, k={K_NEIGHBORS}")
    print(f"Threshold={THRESHOLD}")
    print("=" * 80)

    final_pipeline = build_final_pipeline()

    # Fit ONCE on the complete development cohort.
    final_pipeline.fit(X_train, y_train)

    p_train = final_pipeline.predict_proba(X_train)[:, 1]
    p_ext = final_pipeline.predict_proba(X_ext)[:, 1]

    train_metrics = calc_metrics(y_train, p_train)
    ext_metrics = calc_metrics(y_ext, p_ext)
    ext_ci = bootstrap_external(y_ext, p_ext)

    (
        oof_summary,
        oof_calibration,
        oof_probability_bands,
        oof_candidate_thresholds,
        oof_dca,
        oof_predictions,
    ) = make_oof_results(X_train, y_train)

    # Added outputs for probability interpretation / future app risk display.
    # These do NOT change the model, resampling, parameters, or the locked 0.20 threshold.
    calibration_summary = pd.concat([
        make_calibration_summary(y_train, p_train, "Full_Training"),
        make_calibration_summary(y_ext, p_ext, "External_Test"),
    ], ignore_index=True)

    probability_bands = pd.concat([
        make_probability_bins(y_train, p_train, "Full_Training"),
        make_probability_bins(y_ext, p_ext, "External_Test"),
    ], ignore_index=True)

    candidate_thresholds = pd.concat([
        make_candidate_threshold_table(y_train, p_train, "Full_Training"),
        make_candidate_threshold_table(y_ext, p_ext, "External_Test"),
    ], ignore_index=True)

    dca_external = make_dca_table(y_ext, p_ext, "External_Test")

    predictions = pd.concat([
        pd.DataFrame({
            "Dataset": "Full_Training",
            "Observed": y_train,
            "Probability": p_train,
            "Prediction": (p_train >= THRESHOLD).astype(int),
        }),
        pd.DataFrame({
            "Dataset": "External_Test",
            "Observed": y_ext,
            "Probability": p_ext,
            "Prediction": (p_ext >= THRESHOLD).astype(int),
        }),
    ], ignore_index=True)

    setting = pd.DataFrame([{
        "Model": "CatBoost",
        "N_Predictors": X_train.shape[1],
        "Resampling": "SMOTENC",
        "Sampling_Ratio": SAMPLING_RATIO,
        "Threshold": THRESHOLD,
        "Scaler": "Training-fit StandardScaler for continuous predictors; sex unchanged",
        "Iterations": MODEL_PARAMS["iterations"],
        "Learning_Rate": MODEL_PARAMS["learning_rate"],
        "Depth": MODEL_PARAMS["depth"],
        "L2_Leaf_Reg": MODEL_PARAMS["l2_leaf_reg"],
        "Random_Strength": MODEL_PARAMS["random_strength"],
        "Bagging_Temperature": MODEL_PARAMS["bagging_temperature"],
        "External_Used_For_Training": "No",
        "External_Used_For_Tuning": "No",
    }])

    summary = pd.DataFrame([
        {"Dataset": "Full_Training", **train_metrics},
        {"Dataset": "External_Test", **ext_metrics},
    ])

    with pd.ExcelWriter(
        os.path.join(output_dir, "01_final_CatBoost_results.xlsx"),
        engine="openpyxl",
    ) as writer:
        setting.to_excel(writer, index=False, sheet_name="Final_Setting")
        summary.to_excel(writer, index=False, sheet_name="Point_Estimates")
        ext_ci.to_excel(writer, index=False, sheet_name="External_95CI")
        predictions.to_excel(writer, index=False, sheet_name="Predictions")
        calibration_summary.to_excel(writer, index=False, sheet_name="Calibration_Summary")
        probability_bands.to_excel(writer, index=False, sheet_name="Probability_Bands")
        candidate_thresholds.to_excel(writer, index=False, sheet_name="Candidate_Thresholds")
        dca_external.to_excel(writer, index=False, sheet_name="External_DCA")

        oof_sheet = "OOF_Results"
        row = 0

        pd.DataFrame({"Section": ["OOF SUMMARY METRICS"]}).to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += 2
        oof_summary.to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += len(oof_summary) + 3

        pd.DataFrame({"Section": ["OOF CALIBRATION SUMMARY"]}).to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += 2
        oof_calibration.to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += len(oof_calibration) + 3

        pd.DataFrame({"Section": ["OOF PROBABILITY BANDS"]}).to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += 2
        oof_probability_bands.to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += len(oof_probability_bands) + 3

        pd.DataFrame({"Section": ["OOF CANDIDATE THRESHOLDS"]}).to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += 2
        oof_candidate_thresholds.to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += len(oof_candidate_thresholds) + 3

        pd.DataFrame({"Section": ["OOF DECISION CURVE TABLE"]}).to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += 2
        oof_dca.to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += len(oof_dca) + 3

        pd.DataFrame({"Section": ["OOF PATIENT-LEVEL PREDICTIONS"]}).to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )
        row += 2
        oof_predictions.to_excel(
            writer, index=False, sheet_name=oof_sheet, startrow=row
        )

    save_scaler(final_pipeline, X_train, output_dir)
    save_plots(y_train, p_train, y_ext, p_ext, output_dir)
    save_added_plots(y_train, p_train, y_ext, p_ext, dca_external, output_dir)

    print("FINISHED")
    print("Output:", output_dir)
    print(summary.to_string(index=False))
    print("\nOOF 5-fold summary:")
    print(oof_summary.to_string(index=False))


if __name__ == "__main__":
    main()
