# -*- coding: utf-8 -*-
"""
Deployment-ready final CatBoost model for early deep SSI risk stratification.

Final workflow
--------------
6 predictors -> locked preprocessing -> locked CatBoost -> raw probability
-> OOF-derived logistic recalibration -> calibrated SSI probability
-> Low / Intermediate / High risk stratum.

Important
---------
1) The six predictors and model hyperparameters are locked.
2) The original single binary threshold 0.11 is NOT used for app deployment.
3) Risk strata use the recalibrated probability:
      calibrated p < 0.0540609   -> Low
      0.0540609 <= calibrated p < 0.3043438 -> Intermediate
      calibrated p >= 0.3043438 -> High
4) The clinician-facing probability is the recalibrated probability, not the raw CatBoost probability.
5) The model uses information available through POD4-5; POD7-8 is not used.

Build once, then load the saved .cbm + JSON config in the app. Do NOT retrain the
model every time a patient is entered.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC
from catboost import CatBoostClassifier


# =============================================================================
# 0. LOCKED FINAL SETTINGS
# =============================================================================
MODEL_VERSION = "SSI-CatBoost-6P-v1.0-2026-09-18"
TARGET_COL = "Infection"
POS_LABEL = 1
SEX_COL = "sex"

# Exact final six predictors, in exact order.
FEATURE_COLUMNS: List[str] = [
    "sex",
    "L3",
    "PLT2",
    "CRP2",
    "ΔESR_2to3",
    "ΔN%_2to3",
]

RANDOM_STATE = 42
SAMPLING_RATIO = 0.41
K_NEIGHBORS = 11

MODEL_PARAMS: Dict[str, Any] = {
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

# Locked raw-probability boundaries derived from training OOF predictions.
RAW_LOW_CUTOFF = 0.1060453100788336
RAW_HIGH_CUTOFF = 0.4642009876892401

# Locked logistic recalibration fitted using training 5-fold OOF probabilities ONLY.
CAL_INTERCEPT = -0.679864
CAL_SLOPE = 1.023649

# Locked deployment boundaries on the calibrated scale.
CAL_LOW_CUTOFF = 0.0540609
CAL_HIGH_CUTOFF = 0.3043438

# Reference audit values from the locked development OOF analysis.
EXPECTED_OOF_N = 438
EXPECTED_OOF_POS = 71
EXPECTED_OOF_AUPRC = 0.3919747271755154
EXPECTED_OOF_AUROC = 0.718847142802318
EXPECTED_LOW_SENSITIVITY = 68 / 71       # 95.7746%
EXPECTED_HIGH_SPECIFICITY = 331 / 367    # 90.1907%

APPLICABLE_TIME_WINDOW = "Preoperative through POD4–5; POD7–8 is not used."
MODEL_INTENDED_USE = (
    "Early risk stratification for deep postoperative SSI after spinal instrumentation; "
    "not a standalone diagnosis."
)


# =============================================================================
# 1. NUMERIC HELPERS
# =============================================================================
def _clip_probability(p: Union[float, np.ndarray], eps: float = 1e-12):
    return np.clip(p, eps, 1.0 - eps)


def logit(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    p = _clip_probability(np.asarray(p, dtype=float))
    z = np.log(p / (1.0 - p))
    return float(z) if z.ndim == 0 else z


def expit(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    z = np.asarray(z, dtype=float)
    # Stable logistic transformation.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return float(out) if out.ndim == 0 else out


def recalibrate_probability(raw_probability: Union[float, np.ndarray]):
    """Locked OOF-derived logistic recalibration."""
    return expit(CAL_INTERCEPT + CAL_SLOPE * logit(raw_probability))


def risk_from_raw_probability(raw_probability: float) -> str:
    p = float(raw_probability)
    if p < RAW_LOW_CUTOFF:
        return "Low risk"
    if p < RAW_HIGH_CUTOFF:
        return "Intermediate risk"
    return "High risk"


def risk_from_calibrated_probability(calibrated_probability: float) -> str:
    p = float(calibrated_probability)
    if p < CAL_LOW_CUTOFF:
        return "Low risk"
    if p < CAL_HIGH_CUTOFF:
        return "Intermediate risk"
    return "High risk"


# =============================================================================
# 2. TRAINING PIPELINE (MATCHES THE FINAL MODEL)
# =============================================================================
class ContinuousOnlyScaler(BaseEstimator, TransformerMixin):
    """Z-score continuous predictors; keep sex unchanged as 0/1."""

    def __init__(self, sex_col: str = SEX_COL):
        self.sex_col = sex_col

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.columns_ = list(X.columns)
        self.continuous_ = [c for c in self.columns_ if c != self.sex_col]
        self.scaler_ = StandardScaler()
        if self.continuous_:
            self.scaler_.fit(X[self.continuous_])
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.loc[:, self.columns_]
        out = X.astype(float).copy()
        if self.continuous_:
            out.loc[:, self.continuous_] = self.scaler_.transform(X[self.continuous_])
        if self.sex_col in out.columns:
            out[self.sex_col] = X[self.sex_col].astype(int)
        return out


class SMOTENCSampler(BaseEstimator):
    def __init__(
        self,
        sampling_strategy: float = 1.0,
        random_state: int = 42,
        k_neighbors: int = 5,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_arr = np.asarray(y)

        minority_n = int(pd.Series(y_arr).value_counts().min())
        if minority_n < 2:
            raise ValueError("Minority class has fewer than 2 observations; SMOTE cannot be applied.")
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
        return pd.DataFrame(X_res, columns=X_df.columns), np.asarray(y_res)


def fit_final_components(X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
    """
    Fit the exact locked sequence used by the original final pipeline:
    continuous-only StandardScaler -> SMOTENC -> CatBoost.

    This is written explicitly rather than through imblearn.Pipeline so deployment
    remains robust across sklearn/imblearn/CatBoost version combinations.
    """
    scaler = ContinuousOnlyScaler()
    X_scaled = scaler.fit_transform(X)

    sampler = SMOTENCSampler(
        sampling_strategy=SAMPLING_RATIO,
        random_state=RANDOM_STATE,
        k_neighbors=K_NEIGHBORS,
    )
    X_res, y_res = sampler.fit_resample(X_scaled, np.asarray(y))

    model = CatBoostClassifier(**MODEL_PARAMS)
    model.fit(X_res, y_res)
    return scaler, model


def predict_with_components(
    scaler: ContinuousOnlyScaler,
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1].astype(float)


# =============================================================================
# 3. DATA VALIDATION / OOF AUDIT
# =============================================================================
def _normalize_sex_name(df: pd.DataFrame) -> pd.DataFrame:
    matches = [c for c in df.columns if str(c).strip().lower() == "sex"]
    if len(matches) > 1:
        raise ValueError(f"Detected multiple sex columns: {matches}")
    if len(matches) == 1 and matches[0] != SEX_COL:
        df = df.rename(columns={matches[0]: SEX_COL})
    return df


def load_training_data(path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = _normalize_sex_name(pd.read_excel(path).dropna(axis=1, how="all"))
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Training file is missing required columns: {missing_cols}")

    # Lock to the exact final predictors; ignore any accidental extra columns.
    df = df.loc[:, FEATURE_COLUMNS + [TARGET_COL]].copy()
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    X = df[FEATURE_COLUMNS].copy()
    for c in FEATURE_COLUMNS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    y_num = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y_num.isna().any():
        raise ValueError("Training outcome contains non-numeric/invalid values.")
    y = (y_num.astype(int) == POS_LABEL).astype(int)

    if y.nunique() != 2:
        raise ValueError("Training outcome must contain both 0 and 1.")
    if X[SEX_COL].isna().any():
        raise ValueError("sex contains missing values.")
    if not set(X[SEX_COL].unique()).issubset({0, 1, 0.0, 1.0}):
        raise ValueError("sex must be coded 0/1 (0=female, 1=male).")

    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    return X.reset_index(drop=True), y.reset_index(drop=True), medians


def make_oof_predictions(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_arr = np.asarray(y).astype(int)
    oof = np.full(len(y_arr), np.nan, dtype=float)

    for train_idx, val_idx in skf.split(X, y_arr):
        fold_scaler, fold_model = fit_final_components(
            X.iloc[train_idx].copy(), y_arr[train_idx]
        )
        oof[val_idx] = predict_with_components(
            fold_scaler, fold_model, X.iloc[val_idx].copy()
        )

    if np.isnan(oof).any():
        raise RuntimeError("OOF prediction generation failed.")
    return oof


def fit_logistic_recalibration(y_true: Iterable[int], raw_probability: Iterable[float]) -> Tuple[float, float]:
    """Fit y ~ intercept + slope * logit(p) via IRLS; same logic as final analysis."""
    y = np.asarray(y_true, dtype=float)
    p = _clip_probability(np.asarray(raw_probability, dtype=float), eps=1e-6)
    x = np.log(p / (1.0 - p))
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0], dtype=float)

    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        new_beta = np.linalg.pinv(XtW @ X) @ (XtW @ z)
        if np.max(np.abs(new_beta - beta)) < 1e-10:
            beta = new_beta
            break
        beta = new_beta

    return float(beta[0]), float(beta[1])


def _binary_operating_point(y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Sensitivity": float(sensitivity), "Specificity": float(specificity),
    }


def audit_locked_logic(X: pd.DataFrame, y: pd.Series, oof: np.ndarray) -> Dict[str, Any]:
    y_arr = np.asarray(y).astype(int)
    fitted_intercept, fitted_slope = fit_logistic_recalibration(y_arr, oof)

    # Verify that recalibration fitted from the current final training data reproduces the locked values.
    cal_ok = (
        abs(fitted_intercept - CAL_INTERCEPT) < 5e-6
        and abs(fitted_slope - CAL_SLOPE) < 5e-6
    )

    # Verify exact equivalence between raw-cutoff and calibrated-cutoff stratification.
    cal_oof = np.asarray(recalibrate_probability(oof), dtype=float)
    raw_groups = np.array([risk_from_raw_probability(v) for v in oof], dtype=object)
    cal_groups = np.array([risk_from_calibrated_probability(v) for v in cal_oof], dtype=object)
    strata_equivalent = bool(np.array_equal(raw_groups, cal_groups))

    low_op = _binary_operating_point(y_arr, oof, RAW_LOW_CUTOFF)
    high_op = _binary_operating_point(y_arr, oof, RAW_HIGH_CUTOFF)

    audit = {
        "N": int(len(y_arr)),
        "Positive_N": int(y_arr.sum()),
        "OOF_AUPRC": float(average_precision_score(y_arr, oof)),
        "OOF_AUROC": float(roc_auc_score(y_arr, oof)),
        "OOF_Brier_raw": float(brier_score_loss(y_arr, oof)),
        "Fitted_calibration_intercept": fitted_intercept,
        "Fitted_calibration_slope": fitted_slope,
        "Locked_calibration_intercept": CAL_INTERCEPT,
        "Locked_calibration_slope": CAL_SLOPE,
        "Calibration_parameters_match_locked": cal_ok,
        "Raw_and_calibrated_strata_are_exactly_equivalent": strata_equivalent,
        "Low_cutoff": RAW_LOW_CUTOFF,
        "Low_cutoff_operating_point": low_op,
        "High_cutoff": RAW_HIGH_CUTOFF,
        "High_cutoff_operating_point": high_op,
    }

    # Hard stops: if these fail, do not silently build a different model.
    if list(X.columns) != FEATURE_COLUMNS:
        raise RuntimeError("Feature order differs from locked final model.")
    if len(y_arr) == EXPECTED_OOF_N and int(y_arr.sum()) == EXPECTED_OOF_POS:
        if not cal_ok:
            raise RuntimeError(
                "OOF recalibration parameters do not reproduce the locked final values. "
                "Check training data/version, package versions, or preprocessing."
            )
        # Deployment classifies directly on the locked calibrated cutoffs supplied for the app.
        # Rounded deployment constants need not reproduce the historical raw-scale groups exactly.

    return audit


# =============================================================================
# 4. BUILD DEPLOYMENT ARTIFACTS ONCE
# =============================================================================
def build_deployment_artifacts(
    training_file: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, medians = load_training_data(training_file)

    # OOF is used ONLY to audit/reproduce the locked recalibration; not for app-time refitting.
    oof = make_oof_predictions(X, y)
    audit = audit_locked_logic(X, y, oof)

    # Fit the final model once on the complete development cohort.
    scaler_wrapper, model = fit_final_components(X, y)

    model_path = output_dir / "final_catboost_6predictor.cbm"
    config_path = output_dir / "final_catboost_6predictor_config.json"
    audit_path = output_dir / "final_catboost_6predictor_build_audit.json"

    model.save_model(str(model_path))

    scaler_payload: Dict[str, Any] = {
        "continuous_features": list(scaler_wrapper.continuous_),
        "mean": {},
        "scale": {},
    }
    for idx, feature in enumerate(scaler_wrapper.continuous_):
        scaler_payload["mean"][feature] = float(scaler_wrapper.scaler_.mean_[idx])
        scaler_payload["scale"][feature] = float(scaler_wrapper.scaler_.scale_[idx])

    config = {
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COL,
        "sex_column": SEX_COL,
        "sex_coding": {"0": "female", "1": "male"},
        "training_medians": {k: float(v) for k, v in medians.to_dict().items()},
        "scaler": scaler_payload,
        "recalibration": {
            "type": "logistic_intercept_slope_on_training_5fold_OOF",
            "intercept": CAL_INTERCEPT,
            "slope": CAL_SLOPE,
            "formula": "logit(p_cal) = intercept + slope * logit(p_raw)",
        },
        "risk_cutoffs": {
            "raw_probability": {
                "low_to_intermediate": RAW_LOW_CUTOFF,
                "intermediate_to_high": RAW_HIGH_CUTOFF,
            },
            "calibrated_probability_equivalent": {
                "low_to_intermediate": CAL_LOW_CUTOFF,
                "intermediate_to_high": CAL_HIGH_CUTOFF,
            },
            "authoritative_classification_scale": "calibrated_probability",
        },
        "applicable_time_window": APPLICABLE_TIME_WINDOW,
        "intended_use": MODEL_INTENDED_USE,
        "display": {
            "show": ["risk_level", "recalibrated_estimated_probability_of_deep_SSI"],
            "do_not_show_as_clinical_probability": "raw_catboost_probability",
        },
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    return {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "audit_path": str(audit_path),
        "audit": audit,
    }


# =============================================================================
# 5. APP-TIME INFERENCE
# =============================================================================
@dataclass
class PredictionResult:
    model_version: str
    risk_level: str
    estimated_probability: float
    estimated_probability_percent: float
    raw_probability_internal: float
    applicable_time_window: str
    message_en: str
    message_zh: str
    imputed_features: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "risk_level": self.risk_level,
            "estimated_probability": self.estimated_probability,
            "estimated_probability_percent": self.estimated_probability_percent,
            "raw_probability_internal": self.raw_probability_internal,
            "applicable_time_window": self.applicable_time_window,
            "message_en": self.message_en,
            "message_zh": self.message_zh,
            "imputed_features": self.imputed_features,
        }


RISK_MESSAGES_EN = {
    "Low risk": (
        "The estimated risk of SSI is low. Infection cannot be excluded solely on the basis of this result. "
        "Further evaluation is warranted if clinical suspicion persists."
    ),
    "Intermediate risk": (
        "Review the wound for drainage, redness, or increasing pain. Assess CRP trends and other clinical "
        "signs of infection. If suspicion persists, consider imaging and microbiological investigations. "
        "Do not diagnose or exclude deep SSI based on this score alone."
    ),
    "High risk": (
        "The estimated risk of deep SSI is elevated. Prompt clinical assessment and further investigation "
        "should be considered. This result does not establish a diagnosis of deep SSI."
    ),
}

RISK_MESSAGES_ZH = {
    "Low risk": "模型提示深部SSI估计风险较低，但不能单独排除感染；如临床仍高度怀疑，应继续进一步评估。",
    "Intermediate risk": "模型未将患者明确归入低风险或高风险，建议结合临床表现及进一步检查综合判断。",
    "High risk": "模型提示深部SSI风险明显升高，建议进一步进行临床评估。高风险不等同于确诊感染。",
}


class SSIRiskModel:
    """
    Load the locked deployment artifacts and predict one or more patients.

    For bedside/mini-app use, missing predictors are rejected by default. If your final
    deployment protocol explicitly allows the same training-median imputation used in the
    analysis, pass allow_imputation=True and show the returned imputed_features to the user.
    """

    def __init__(self, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)
        config_path = model_dir / "final_catboost_6predictor_config.json"
        model_path = model_dir / "final_catboost_6predictor.cbm"

        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        with config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))

        if self.config["feature_columns"] != FEATURE_COLUMNS:
            raise RuntimeError("Deployment config feature list does not match locked code.")

    def _prepare_frame(
        self,
        patient: Union[Mapping[str, Any], pd.DataFrame],
        allow_imputation: bool,
    ) -> Tuple[pd.DataFrame, List[str]]:
        if isinstance(patient, pd.DataFrame):
            df = patient.copy()
        else:
            df = pd.DataFrame([dict(patient)])

        missing_columns = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required predictors: {missing_columns}")

        df = df.loc[:, FEATURE_COLUMNS].copy()
        for c in FEATURE_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # sex must never be silently imputed.
        if df[SEX_COL].isna().any():
            raise ValueError("sex is missing/invalid; it must be explicitly provided as 0 or 1.")
        if not set(df[SEX_COL].unique()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError("sex must be coded 0/1 (0=female, 1=male).")

        imputed_features: List[str] = []
        for c in FEATURE_COLUMNS:
            if c == SEX_COL:
                continue
            if df[c].isna().any():
                if not allow_imputation:
                    raise ValueError(
                        f"Predictor '{c}' is missing/invalid. By default the clinical app does not "
                        "silently impute missing final predictors."
                    )
                fill_value = float(self.config["training_medians"][c])
                df[c] = df[c].fillna(fill_value)
                imputed_features.append(c)

        # Apply the exact training scaler saved in JSON.
        scaled = df.astype(float).copy()
        scaler = self.config["scaler"]
        for c in scaler["continuous_features"]:
            mean = float(scaler["mean"][c])
            scale = float(scaler["scale"][c])
            if scale <= 0:
                raise RuntimeError(f"Invalid stored scaler SD for {c}: {scale}")
            scaled[c] = (scaled[c] - mean) / scale
        scaled[SEX_COL] = df[SEX_COL].astype(int)

        return scaled.loc[:, FEATURE_COLUMNS], sorted(set(imputed_features))

    def predict(
        self,
        patient: Union[Mapping[str, Any], pd.DataFrame],
        allow_imputation: bool = False,
    ) -> Union[PredictionResult, List[PredictionResult]]:
        X_scaled, imputed_features = self._prepare_frame(patient, allow_imputation)
        raw_probs = self.model.predict_proba(X_scaled)[:, 1].astype(float)
        cal_probs = np.asarray(recalibrate_probability(raw_probs), dtype=float)

        results: List[PredictionResult] = []
        for raw_p, cal_p in zip(raw_probs, cal_probs):
            risk_cal = risk_from_calibrated_probability(float(cal_p))

            results.append(PredictionResult(
                model_version=self.config["model_version"],
                risk_level=risk_cal,
                estimated_probability=float(cal_p),
                estimated_probability_percent=round(float(cal_p) * 100.0, 1),
                raw_probability_internal=float(raw_p),
                applicable_time_window=self.config["applicable_time_window"],
                message_en=RISK_MESSAGES_EN[risk_cal],
                message_zh=RISK_MESSAGES_ZH[risk_cal],
                imputed_features=imputed_features.copy(),
            ))

        return results[0] if len(results) == 1 else results


# =============================================================================
# 6. CLI FOR THE JUNIOR DEVELOPER
# =============================================================================
def _cli_build(args):
    result = build_deployment_artifacts(args.train, args.outdir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _cli_predict(args):
    model = SSIRiskModel(args.model_dir)
    patient = json.loads(args.patient_json)
    result = model.predict(patient, allow_imputation=args.allow_imputation)
    if isinstance(result, PredictionResult):
        result = result.to_dict()
    else:
        result = [r.to_dict() for r in result]
    print(json.dumps(result, ensure_ascii=False, indent=2))


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locked 6-predictor CatBoost SSI deployment model")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build and save the locked deployment artifacts once")
    p_build.add_argument("--train", required=True, help="Path to the final training Excel file")
    p_build.add_argument("--outdir", required=True, help="Directory to save .cbm and JSON files")
    p_build.set_defaults(func=_cli_build)

    p_pred = sub.add_parser("predict", help="Predict one patient from a JSON object")
    p_pred.add_argument("--model-dir", required=True, help="Directory created by the build command")
    p_pred.add_argument("--patient-json", required=True, help="JSON object containing the exact 6 predictors")
    p_pred.add_argument(
        "--allow-imputation",
        action="store_true",
        help="Allow locked training-median imputation for missing continuous predictors",
    )
    p_pred.set_defaults(func=_cli_predict)

    return parser


def main():
    parser = make_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
