"""
Модели прогнозирования матчей.
Ensemble: XGBoost + LogisticRegression + калибровка Platt.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (brier_score_loss, log_loss,
                             accuracy_score, roc_auc_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from data.features import feature_cols

MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(exist_ok=True)


def build_ensemble():
    estimators = [
        ("lr", LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")),
        ("gb", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42)),
    ]
    if HAS_XGB:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42)))

    voting = VotingClassifier(estimators, voting="soft")
    # Platt-scaling для честной калибровки вероятностей
    return CalibratedClassifierCV(voting, cv=3, method="sigmoid")


class MatchPredictor:
    def __init__(self):
        self.model = build_ensemble()
        self.scaler = StandardScaler()
        self.fcols: Optional[list] = None
        self.trained = False

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.fcols].fillna(0).values
        return self.scaler.transform(X)

    def train(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Walk-forward кросс-валидация (TimeSeriesSplit=5),
        затем финальное обучение на всём датасете.
        """
        self.fcols = feature_cols(df)
        X = df[self.fcols].fillna(0).values
        y = df["label"].values

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = {"accuracy": [], "auc": [], "brier": [], "logloss": []}

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr_idx])
            X_val = sc.transform(X[val_idx])

            m = build_ensemble()
            m.fit(X_tr, y[tr_idx])
            proba = m.predict_proba(X_val)[:, 1]
            pred  = (proba >= 0.5).astype(int)

            cv_scores["accuracy"].append(accuracy_score(y[val_idx], pred))
            cv_scores["auc"].append(roc_auc_score(y[val_idx], proba))
            cv_scores["brier"].append(brier_score_loss(y[val_idx], proba))
            cv_scores["logloss"].append(log_loss(y[val_idx], proba))

            if verbose:
                print(f"  Fold {fold+1}: acc={cv_scores['accuracy'][-1]:.3f}  "
                      f"AUC={cv_scores['auc'][-1]:.3f}  "
                      f"Brier={cv_scores['brier'][-1]:.3f}")

        # Финальное обучение на 100% данных
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.trained = True

        summary = {k: float(np.mean(v)) for k, v in cv_scores.items()}
        if verbose:
            print(f"\n[CV среднее]  acc={summary['accuracy']:.3f}  "
                  f"AUC={summary['auc']:.3f}  Brier={summary['brier']:.3f}  "
                  f"LogLoss={summary['logloss']:.3f}")
        return summary

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает DataFrame с колонками prob1, prob2, edge, kelly."""
        assert self.trained, "Сначала обучи модель: predictor.train(df)"
        proba = self.model.predict_proba(self._X(df))[:, 1]
        out = df[["match_id", "date", "team1_name", "team2_name"]].copy()
        out["prob1"] = proba.round(4)
        out["prob2"] = (1 - proba).round(4)
        return out

    def save(self, name: str = "model"):
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "fcols": self.fcols}, f)
        print(f"[✓] Модель сохранена → {path}")

    def load(self, name: str = "model"):
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.model, self.scaler, self.fcols = obj["model"], obj["scaler"], obj["fcols"]
        self.trained = True
        print(f"[✓] Модель загружена ← {path}")
