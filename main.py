#!/usr/bin/env python3
"""
Esports Match Predictor — CLI
═══════════════════════════════════════════════════
Команды:
  python main.py fetch   --game csgo --days 180
  python main.py train   --game csgo
  python main.py predict --game csgo --team1 "NaVi" --team2 "Astralis"
  python main.py backtest --game csgo
═══════════════════════════════════════════════════
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from data.collector import DataCollector
from data.features import build_features
from models.predictor import MatchPredictor
from backtest.runner import walk_forward_backtest
from utils.bankroll import value_edge, kelly_fraction


def cmd_fetch(args):
    token = os.getenv("PANDASCORE_TOKEN")
    collector = DataCollector(api_token=token)
    collector.fetch_matches(game=args.game, days_back=args.days)


def cmd_train(args):
    df_raw = DataCollector.load_local(game=args.game)
    df = build_features(df_raw)
    pred = MatchPredictor()
    pred.train(df, verbose=True)
    pred.save(name=args.game)


def cmd_predict(args):
    """
    Прогноз для конкретного матча.
    Передаёшь названия команд + (опционально) коэффициенты.
    """
    import pandas as pd
    from data.features import compute_elo, feature_cols

    df_raw = DataCollector.load_local(game=args.game)
    df = build_features(df_raw)

    pred = MatchPredictor()
    try:
        pred.load(name=args.game)
    except FileNotFoundError:
        print("[!] Модель не найдена. Запусти: python main.py train --game", args.game)
        return

    # Ищем команды в истории
    t1_rows = df[df["team1_name"].str.contains(args.team1, case=False, na=False) |
                 df["team2_name"].str.contains(args.team1, case=False, na=False)]
    t2_rows = df[df["team1_name"].str.contains(args.team2, case=False, na=False) |
                 df["team2_name"].str.contains(args.team2, case=False, na=False)]

    if t1_rows.empty or t2_rows.empty:
        print(f"[!] Команда не найдена в датасете. Доступные: {list(df['team1_name'].unique()[:20])}")
        return

    # Берём последний матч каждой команды как прокси для их текущих признаков
    last1 = t1_rows.iloc[-1]
    last2 = t2_rows.iloc[-1]

    # Собираем синтетическую строку
    fcols = pred.fcols
    synth = {}
    for col in fcols:
        if col.endswith("1") or "1_" in col:
            synth[col] = last1.get(col, 0)
        elif col.endswith("2") or "2_" in col:
            synth[col] = last2.get(col, 0)
        else:
            synth[col] = (last1.get(col, 0) + last2.get(col, 0)) / 2

    import numpy as np
    X = np.nan_to_num(np.array([[synth.get(c, 0) for c in fcols]]), nan=0.0)
    X_sc = pred.scaler.transform(X)
    prob1 = float(pred.model.predict_proba(X_sc)[0, 1])
    prob2 = 1 - prob1

    print(f"\n{'═'*50}")
    print(f"  {args.team1}  vs  {args.team2}")
    print(f"{'═'*50}")
    print(f"  P({args.team1} win) = {prob1:.1%}")
    print(f"  P({args.team2} win) = {prob2:.1%}")

    if args.odds1 and args.odds2:
        odds1, odds2 = float(args.odds1), float(args.odds2)
        e1 = value_edge(prob1, odds1)
        e2 = value_edge(prob2, odds2)
        k1 = kelly_fraction(prob1, odds1)
        k2 = kelly_fraction(prob2, odds2)
        print(f"\n  Коэффициенты:  {args.team1} @ {odds1}  |  {args.team2} @ {odds2}")
        print(f"  Value Edge:    {args.team1}: {e1:+.2%}  |  {args.team2}: {e2:+.2%}")

        best = args.team1 if e1 > e2 else args.team2
        best_k = k1 if e1 > e2 else k2
        best_e = max(e1, e2)

        if best_e > 0:
            print(f"\n  ✅ СТАВКА:  {best}  (Kelly доля: {best_k:.2%} банкролла)")
        else:
            print(f"\n  ❌ Нет value — ставка не рекомендована")
    print(f"{'═'*50}\n")


def cmd_backtest(args):
    df_raw = DataCollector.load_local(game=args.game)
    walk_forward_backtest(df_raw,
                          train_months=args.train_months,
                          initial_bankroll=args.bankroll,
                          kelly_fraction=args.kelly,
                          min_edge=args.min_edge)


def main():
    parser = argparse.ArgumentParser(prog="predictor",
                                     description="Esports Match Predictor")
    sub = parser.add_subparsers(dest="cmd")

    # fetch
    p = sub.add_parser("fetch", help="Загрузить матчи с PandaScore API")
    p.add_argument("--game", default="csgo")
    p.add_argument("--days", type=int, default=180)

    # train
    p = sub.add_parser("train", help="Обучить модель")
    p.add_argument("--game", default="csgo")

    # predict
    p = sub.add_parser("predict", help="Прогноз матча")
    p.add_argument("--game", default="csgo")
    p.add_argument("--team1", required=True)
    p.add_argument("--team2", required=True)
    p.add_argument("--odds1", default=None, help="Коэффициент на team1")
    p.add_argument("--odds2", default=None, help="Коэффициент на team2")

    # backtest
    p = sub.add_parser("backtest", help="Walk-forward бэктест")
    p.add_argument("--game", default="csgo")
    p.add_argument("--train-months", type=int, default=6)
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--kelly", type=float, default=0.25,
                   help="Дробный Kelly (0.25 = четверть Kelly)")
    p.add_argument("--min-edge", type=float, default=0.03,
                   help="Минимальный value edge для ставки")

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    {"fetch": cmd_fetch, "train": cmd_train,
     "predict": cmd_predict, "backtest": cmd_backtest}[args.cmd](args)


if __name__ == "__main__":
    main()
