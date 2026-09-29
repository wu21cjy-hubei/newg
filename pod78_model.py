"""Inference for the POD7–8 CatBoost model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

FEATURES = ["sex", "L3", "N%4", "PLT2", "ΔPLT_1to2", "ΔCRP_2to4", "ΔM%_3to4"]
MESSAGES = {
    "Low risk": "The estimated risk of SSI is low. Infection cannot be excluded solely on the basis of this result. Further evaluation is warranted if clinical suspicion persists.",
    "High risk": "The estimated risk of deep SSI is elevated. Prompt clinical assessment and further investigation should be considered. This result does not establish a diagnosis of deep SSI.",
}


def calibrate(p, intercept, slope):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    z = intercept + slope * (np.log(p) - np.log1p(-p))
    return np.exp(-np.logaddexp(0, -z))


class POD78Model:
    def __init__(self, directory=None):
        directory = Path(directory or Path(__file__).resolve().parent)
        self.config = json.loads((directory / "final_catboost_pod78_config.json").read_text())
        if self.config["feature_columns"] != FEATURES:
            raise ValueError("Model feature order does not match the application.")
        self.model = CatBoostClassifier()
        self.model.load_model(str(directory / "final_catboost_pod78.cbm"))

    def raw_probabilities(self, data):
        frame = pd.DataFrame(data).loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Please enter a finite number for every variable.")
        if not frame["sex"].isin([0, 1]).all():
            raise ValueError("Please select Female or Male.")
        scaled = frame.astype(float).copy()
        scaler = self.config["scaler"]
        for name in scaler["continuous_features"]:
            scaled[name] = (scaled[name] - scaler["mean"][name]) / scaler["scale"][name]
        return self.model.predict_proba(scaled)[:, 1]

    def predict(self, patient):
        raw = float(self.raw_probabilities([patient])[0])
        calibration = self.config["recalibration"]
        probability = float(calibrate(raw, calibration["intercept"], calibration["slope"]))
        risk = "High risk" if raw >= self.config["raw_probability_threshold"] else "Low risk"
        return {"risk_level": risk, "estimated_probability": probability, "message": MESSAGES[risk]}
