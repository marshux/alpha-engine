"""LightGBM-based ML alpha signal: predicts forward excess return vs a
benchmark, rank-transformed for robustness (guide section 3.5's IC-
optimization rationale -- predicting cross-sectional rank is more robust
than predicting raw returns on noisy financial data). Output is
cross-sectionally z-scored via signals/base.py::zscore, so it slots into
backtest/engine.py's signal_table exactly like every other signal
(`signal_weights.ml_alpha` in strategy.yaml).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from alpha_engine.ml.features import FEATURE_COLUMNS
from alpha_engine.signals.base import zscore


@dataclass
class AlphaModel:
    booster: lgb.Booster | None = None
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    trained_through: str | None = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_child_samples: int = 5,
        trained_through: str | None = None,
    ) -> "AlphaModel":
        """`y`: raw forward excess returns -- rank-transformed internally
        (`y.rank(pct=True)`). `min_child_samples` defaults much lower
        than the guide's illustrative 50: that value was tuned for an
        S&P-500-scale cross-section, this repo's universe is a few dozen
        symbols, so a leaf that small would starve every split.
        """
        X = X[self.feature_columns]
        y_rank = y.rank(pct=True)
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            n_jobs=-1,
            verbosity=-1,
            random_state=42,
        )
        model.fit(X, y_rank)
        self.booster = model.booster_
        self.trained_through = trained_through
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Cross-sectionally z-scored prediction, indexed like `X`. Fails
        loudly on a feature-column mismatch rather than silently
        predicting on misaligned columns (e.g. a model saved before a
        feature was added/removed, loaded and used against the new
        feature builder).
        """
        if self.booster is None:
            raise ValueError("AlphaModel is not trained/loaded")
        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f"missing feature columns for prediction: {sorted(missing)}")
        raw = pd.Series(self.booster.predict(X[self.feature_columns]), index=X.index)
        return zscore(raw)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if self.booster is None:
            raise ValueError("cannot save an untrained AlphaModel")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        _meta_path(path).write_text(
            json.dumps({"feature_columns": self.feature_columns, "trained_through": self.trained_through})
        )

    @classmethod
    def load(cls, path: str | Path) -> "AlphaModel":
        path = Path(path)
        meta = json.loads(_meta_path(path).read_text())
        booster = lgb.Booster(model_file=str(path))
        return cls(
            booster=booster,
            feature_columns=meta["feature_columns"],
            trained_through=meta.get("trained_through"),
        )


def _meta_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + ".meta.json")
