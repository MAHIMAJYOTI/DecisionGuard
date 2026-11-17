"""Save and load a fitted model plus a metadata file a person can read."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_gen.config import project_root


def artifact_dir(model_version: str) -> Path:
    return project_root() / "artifacts" / "models" / model_version


@dataclass
class SavedModel:
    """A fitted model and the column order it was trained with."""

    version: str
    kind: str
    features: list[str]
    pipeline: object

    def predict_pd(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of the bad class. Rows with a missing feature stay NaN."""

        # A missing column is a missing feature, not a crash. The engine turns
        # a missing score into a referral or a challenge instead of an approval.
        values = frame.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce")
        scores = np.full(len(frame), np.nan)
        ready = values.notna().all(axis=1).to_numpy()
        if ready.any():
            scores[ready] = self.pipeline.predict_proba(values.loc[ready])[:, 1]
        return scores


def save_model(model: SavedModel, metadata: dict) -> Path:
    folder = artifact_dir(model.version)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.joblib")
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folder


def load_model(model_version: str) -> SavedModel:
    path = artifact_dir(model_version) / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Train one with `python -m src.models`."
        )
    model = joblib.load(path)
    if not isinstance(model, SavedModel):
        raise TypeError(f"{path} is not a SavedModel.")
    return model


def load_metadata(model_version: str) -> dict:
    path = artifact_dir(model_version) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"No metadata at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))
