"""
Управление банкроллом: Kelly Criterion + варианты (fractional, threshold).
"""

import numpy as np
import pandas as pd
from typing import Optional


def kelly_fraction(prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Дробный Kelly Criterion.
    prob  — наша оценка вероятности победы
    odds  — десятичный коэффициент букмекера (напр. 1.85)
    fraction — насколько агрессивно ставим (0.25 = четверть Kelly)
    
    Возвращает долю банкролла для ставки (0.0 если нет value).
    """
    b = odds - 1  # чистый выигрыш на единицу ставки
    q = 1 - prob
    k = (b * prob - q) / b
    if k <= 0:
        return 0.0
    return round(k * fraction, 4)


def value_edge(prob: float, odds: float) -> float:
    """
    Value Edge = наша p * odds - 1.
    > 0 → есть value (ожидаемый плюс).
    """
    return round(prob * odds - 1, 4)


def apply_bankroll(df: pd.DataFrame,
                   initial: float = 1000.0,
                   fraction: float = 0.25,
                   min_edge: float = 0.03,
                   max_stake_pct: float = 0.05,
                   odds_col1: str = "odds1",
                   odds_col2: str = "odds2") -> pd.DataFrame:
    """
    Симулирует ставки по стратегии Kelly на историческом датасете.
    Нужны колонки: prob1, prob2, label, odds1, odds2.
    """
    records = []
    bankroll = initial

    for _, row in df.iterrows():
        prob1, prob2 = row["prob1"], row["prob2"]
        odds1 = row.get(odds_col1, np.nan)
        odds2 = row.get(odds_col2, np.nan)
        label = row["label"]

        best_bet = None
        best_edge = min_edge

        if not np.isnan(odds1):
            e1 = value_edge(prob1, odds1)
            if e1 > best_edge:
                best_edge = e1
                best_bet = ("team1", prob1, odds1, e1)

        if not np.isnan(odds2):
            e2 = value_edge(prob2, odds2)
            if e2 > best_edge:
                best_bet = ("team2", prob2, odds2, e2)

        if best_bet is None:
            records.append({**row.to_dict(), "bet_on": None, "stake": 0,
                            "pnl": 0, "bankroll": bankroll})
            continue

        side, prob, odds, edge = best_bet
        k = kelly_fraction(prob, odds, fraction)
        stake = min(bankroll * k, bankroll * max_stake_pct)
        stake = round(stake, 2)

        won = (side == "team1" and label == 1) or (side == "team2" and label == 0)
        pnl = stake * (odds - 1) if won else -stake
        bankroll += pnl

        records.append({**row.to_dict(),
                        "bet_on": side, "stake": stake, "edge": edge,
                        "pnl": round(pnl, 2), "bankroll": round(bankroll, 2)})

    return pd.DataFrame(records)


def bankroll_stats(sim: pd.DataFrame, initial: float = 1000.0) -> dict:
    bets = sim[sim["stake"] > 0]
    if bets.empty:
        return {"error": "Нет ставок"}

    wins = bets[bets["pnl"] > 0]
    total_wagered = bets["stake"].sum()
    total_pnl = bets["pnl"].sum()

    # Max drawdown
    peak = sim["bankroll"].cummax()
    drawdown = (sim["bankroll"] - peak) / peak
    max_dd = drawdown.min()

    return {
        "total_bets":    len(bets),
        "win_rate":      round(len(wins) / len(bets), 3),
        "roi":           round(total_pnl / total_wagered, 4),
        "total_pnl":     round(total_pnl, 2),
        "final_bankroll": round(sim["bankroll"].iloc[-1], 2),
        "growth_pct":    round((sim["bankroll"].iloc[-1] / initial - 1) * 100, 2),
        "max_drawdown":  round(max_dd * 100, 2),
        "avg_edge":      round(bets["edge"].mean(), 4),
    }
