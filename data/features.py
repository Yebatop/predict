"""
Feature Engineering для прогнозирования матчей.
Все признаки считаются строго на исторических данных (no data leakage).
"""

import numpy as np
import pandas as pd
from typing import Tuple


def _elo_update(r_winner: float, r_loser: float, K: float = 32) -> Tuple[float, float]:
    expected = 1 / (1 + 10 ** ((r_loser - r_winner) / 400))
    new_winner = r_winner + K * (1 - expected)
    new_loser  = r_loser  + K * (0 - (1 - expected))
    return new_winner, new_loser


def compute_elo(df: pd.DataFrame, K: float = 32, init: float = 1500.0) -> pd.DataFrame:
    """Считает ELO-рейтинги команд на момент ПЕРЕД каждым матчем."""
    elo: dict = {}
    elo_before_1, elo_before_2 = [], []

    for _, row in df.iterrows():
        t1, t2 = row["team1_id"], row["team2_id"]
        r1 = elo.get(t1, init)
        r2 = elo.get(t2, init)
        elo_before_1.append(r1)
        elo_before_2.append(r2)

        if row["label"] == 1:
            elo[t1], elo[t2] = _elo_update(r1, r2, K)
        else:
            elo[t2], elo[t1] = _elo_update(r2, r1, K)

    df = df.copy()
    df["elo1"] = elo_before_1
    df["elo2"] = elo_before_2
    df["elo_diff"] = df["elo1"] - df["elo2"]
    return df


def _rolling_winrate(df: pd.DataFrame, team_col: str, label_val: int,
                     window: int, min_periods: int = 3) -> pd.Series:
    """Скользящий винрейт команды за последние N матчей."""
    result = pd.Series(np.nan, index=df.index)
    for tid in df[team_col].unique():
        mask = df[team_col] == tid
        sub = df[mask].copy()
        # label_val=1 → team1 wins, label_val=0 → team2 wins
        if team_col == "team1_id":
            wins = (sub["label"] == 1).astype(float)
        else:
            wins = (sub["label"] == 0).astype(float)
        rolled = wins.shift(1).rolling(window, min_periods=min_periods).mean()
        result.loc[sub.index] = rolled.values
    return result


def _h2h_winrate(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """H2H winrate team1 vs team2 за последние window встреч."""
    result = pd.Series(np.nan, index=df.index)
    for i, row in df.iterrows():
        t1, t2 = row["team1_id"], row["team2_id"]
        past = df.loc[:i - 1 if i > 0 else 0]
        past = past[
            ((past["team1_id"] == t1) & (past["team2_id"] == t2)) |
            ((past["team1_id"] == t2) & (past["team2_id"] == t1))
        ].tail(window)
        if len(past) < 2:
            continue
        wins_t1 = ((past["team1_id"] == t1) & (past["label"] == 1)).sum() + \
                  ((past["team2_id"] == t1) & (past["label"] == 0)).sum()
        result.at[i] = wins_t1 / len(past)
    return result


def _days_since_last(df: pd.DataFrame, team_col: str) -> pd.Series:
    """Дней с последнего матча (свежесть / усталость)."""
    result = pd.Series(np.nan, index=df.index)
    last_date: dict = {}
    for i, row in df.iterrows():
        tid = row[team_col]
        if tid in last_date:
            result.at[i] = (row["date"] - last_date[tid]).days
        last_date[tid] = row["date"]
    return result


def build_features(df: pd.DataFrame,
                   form_windows: Tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """
    Собирает все признаки. Возвращает DataFrame с колонками features + label.
    Строки с NaN в ключевых признаках дропаются (начало истории команды).
    """
    df = df.sort_values("date").reset_index(drop=True)
    df = compute_elo(df)

    # Скользящий винрейт
    for w in form_windows:
        df[f"wr1_{w}"] = _rolling_winrate(df, "team1_id", 1, w)
        df[f"wr2_{w}"] = _rolling_winrate(df, "team2_id", 0, w)
        df[f"wr_diff_{w}"] = df[f"wr1_{w}"] - df[f"wr2_{w}"]

    # H2H
    df["h2h_wr"] = _h2h_winrate(df)

    # Дни с последнего матча
    df["rest1"] = _days_since_last(df, "team1_id")
    df["rest2"] = _days_since_last(df, "team2_id")
    df["rest_diff"] = df["rest1"] - df["rest2"]

    # Турнирный тир (примерная прокси — длина серии)
    df["n_games"] = df["n_games"].fillna(1)

    # Убираем матчи без достаточной истории
    key_cols = ["elo_diff", "wr_diff_5", "wr_diff_10", "wr_diff_20"]
    df.dropna(subset=key_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[✓] Features готовы: {len(df)} матчей, {len(feature_cols(df))} признаков")
    return df


def feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in
            ("match_id", "game", "date", "tournament", "series",
             "team1_id", "team1_name", "team2_id", "team2_name",
             "winner_id", "label",
             "elo1", "elo2", "wr1_5", "wr1_10", "wr1_20",
             "wr2_5", "wr2_10", "wr2_20")]
