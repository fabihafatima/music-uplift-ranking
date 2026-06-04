from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from music_uplift_ranking.uplift_models.config import UpliftConfig
from music_uplift_ranking.uplift_models.preprocessing import LATENT_COLS, prepare_features
from music_uplift_ranking.uplift_models.s_learner import SLearner
from music_uplift_ranking.uplift_models.t_learner import TLearner
from music_uplift_ranking.uplift_models.x_learner import XLearner

LEARNERS = {"t_learner": TLearner, "s_learner": SLearner, "x_learner": XLearner}


def train_uplift_models(
    df: pd.DataFrame,
    cfg: UpliftConfig,
    output_dir: str | Path,
    outcome_col: str = "streams",
    primary_learner: str = "x_learner",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "T" not in df.columns:
        raise ValueError("Missing treatment column T")

    tau_true = df["true_ite_streams"].values if "true_ite_streams" in df.columns else None
    y = df[outcome_col].values.astype(float)
    t = df["T"].values.astype(int)
    track_ids = df["track_id"].values

    X, pipe = prepare_features(df, fit=True)

    X_train, X_test, y_train, y_test, t_train, t_test, id_train, id_test, tau_train, tau_test = (
        _split(X, y, t, track_ids, tau_true)
    )

    results = {}
    primary_model = None

    for name, cls in LEARNERS.items():
        model = cls(cfg)
        model.fit(X_train, y_train, t_train)
        out_path = output_dir / f"{name}.joblib"
        joblib.dump({"model": model, "pipeline": pipe, "cfg": cfg}, out_path)
        results[name] = {"path": str(out_path)}
        if name == primary_learner:
            primary_model = model

    # ITE for every track (full catalog scoring)
    X_full, _ = prepare_features(df, fit=False, pipeline=pipe)
    full_preds = primary_model.predict_all(X_full)
    predictions = pd.DataFrame(
        {
            "track_id": track_ids,
            "predicted_lift_score": full_preds["ite"],
            "mu_0": full_preds["mu_0"],
            "mu_1": full_preds["mu_1"],
            "control_outcome_pred": full_preds["mu_0"],
            "treatment_outcome_pred": full_preds["mu_1"],
        }
    )
    predictions.to_parquet(output_dir / "ite_predictions.parquet", index=False)
    joblib.dump(pipe, output_dir / "feature_pipeline.joblib")

    results["primary_learner"] = primary_learner
    results["predictions_path"] = str(output_dir / "ite_predictions.parquet")
    with (output_dir / "training_manifest.json").open("w") as f:
        json.dump(results, f, indent=2)
    return results


def _split(X, y, t, track_ids, tau_true):
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=t)
    tau_train = tau_true[train_idx] if tau_true is not None else None
    tau_test = tau_true[test_idx] if tau_true is not None else None
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        t[train_idx],
        t[test_idx],
        track_ids[train_idx],
        track_ids[test_idx],
        tau_train,
        tau_test,
    )
