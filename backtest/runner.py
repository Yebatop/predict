"""
Бэктестинг: walk-forward симуляция с отчётом.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data.features import build_features, feature_cols
from models.predictor import MatchPredictor
from utils.bankroll import apply_bankroll, bankroll_stats

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def walk_forward_backtest(df_raw: pd.DataFrame,
                          train_months: int = 6,
                          step_months: int = 1,
                          initial_bankroll: float = 1000.0,
                          kelly_fraction: float = 0.25,
                          min_edge: float = 0.03) -> dict:
    """
    Walk-forward бэктест:
    — Обучаемся на train_months месяцах
    — Прогнозируем следующий step_months
    — Сдвигаемся на step_months вперёд
    """
    df = build_features(df_raw)
    df["month"] = df["date"].dt.to_period("M")
    months = df["month"].sort_values().unique()

    if len(months) < train_months + 1:
        raise ValueError(f"Недостаточно данных: {len(months)} мес., нужно >{train_months}")

    all_preds = []

    for i in range(train_months, len(months), step_months):
        train_months_range = months[i - train_months: i]
        test_months_range  = months[i: i + step_months]

        train = df[df["month"].isin(train_months_range)]
        test  = df[df["month"].isin(test_months_range)]

        if len(train) < 50 or len(test) == 0:
            continue

        pred = MatchPredictor()
        pred.train(train, verbose=False)

        preds = pred.predict(test)
        preds["label"] = test["label"].values
        # Добавляем коэффициенты если есть
        for col in ("odds1", "odds2"):
            if col in test.columns:
                preds[col] = test[col].values

        all_preds.append(preds)
        print(f"  [{train_months_range[0]}–{train_months_range[-1]}] "
              f"→ test [{test_months_range[0]}]: {len(test)} матчей")

    if not all_preds:
        raise ValueError("Нет результатов бэктеста")

    combined = pd.concat(all_preds).reset_index(drop=True)
    
    # Симуляция банкролла (только если есть коэффициенты)
    stats = {}
    if "odds1" in combined.columns and "odds2" in combined.columns:
        sim = apply_bankroll(combined, initial=initial_bankroll,
                             fraction=kelly_fraction, min_edge=min_edge)
        stats = bankroll_stats(sim, initial=initial_bankroll)
        _plot_equity(sim, initial_bankroll)
    
    # Accuracy
    acc = (combined["prob1"].round() == combined["label"]).mean()
    stats["accuracy"] = round(float(acc), 3)
    stats["total_matches"] = len(combined)

    _print_report(stats)
    return stats


def _plot_equity(sim: pd.DataFrame, initial: float):
    bets = sim[sim["stake"] > 0].copy()
    if bets.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Equity curve
    axes[0].plot(range(len(sim)), sim["bankroll"], color="#00d4aa", linewidth=1.5)
    axes[0].axhline(initial, color="#666", linestyle="--", linewidth=0.8)
    axes[0].set_title("Equity Curve", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Банкролл")
    axes[0].fill_between(range(len(sim)), sim["bankroll"], initial,
                         where=sim["bankroll"] >= initial,
                         alpha=0.15, color="#00d4aa")
    axes[0].fill_between(range(len(sim)), sim["bankroll"], initial,
                         where=sim["bankroll"] < initial,
                         alpha=0.15, color="#ff4d4d")

    # PnL по ставкам
    colors = ["#00d4aa" if p > 0 else "#ff4d4d" for p in bets["pnl"]]
    axes[1].bar(range(len(bets)), bets["pnl"], color=colors, width=0.8)
    axes[1].axhline(0, color="#666", linewidth=0.8)
    axes[1].set_title("PnL по ставкам", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("PnL")

    plt.tight_layout()
    path = REPORTS_DIR / "equity_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] График сохранён → {path}")


def _print_report(stats: dict):
    print("\n" + "═" * 45)
    print("  РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("═" * 45)
    for k, v in stats.items():
        label = {
            "total_bets":     "Всего ставок",
            "total_matches":  "Всего матчей",
            "win_rate":       "Win Rate",
            "roi":            "ROI",
            "total_pnl":      "Общий PnL",
            "final_bankroll": "Финальный банкролл",
            "growth_pct":     "Рост банкролла %",
            "max_drawdown":   "Max Drawdown %",
            "avg_edge":       "Средний Edge",
            "accuracy":       "Точность прогноза",
        }.get(k, k)
        print(f"  {label:<22} {v}")
    print("═" * 45 + "\n")
