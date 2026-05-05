# Esports Match Predictor

Статистический предиктор исходов киберспортивных матчей.

## Стек
- **Модель**: XGBoost + GradientBoosting + LogisticRegression (soft voting, Platt calibration)
- **Признаки**: ELO-рейтинг, скользящий winrate (5/10/20 матчей), H2H, дни отдыха
- **Кросс-валидация**: TimeSeriesSplit (walk-forward, no data leakage)
- **Банкролл**: Дробный Kelly Criterion с value edge фильтром

## Быстрый старт

```bash
pip install -r requirements.txt

# 1. Задай API токен
echo "PANDASCORE_TOKEN=твой_токен" > .env

# 2. Загрузи данные (CS:GO, 180 дней)
python main.py fetch --game csgo --days 180

# 3. Обучи модель
python main.py train --game csgo

# 4. Прогноз матча
python main.py predict --game csgo --team1 "NaVi" --team2 "Astralis" \
       --odds1 1.85 --odds2 2.10

# 5. Бэктест
python main.py backtest --game csgo --bankroll 1000 --kelly 0.25 --min-edge 0.03
```

## Поддерживаемые игры
`csgo` | `dota2` | `lol` | `valorant`

## Признаки модели
| Признак | Описание |
|---|---|
| `elo_diff` | Разница ELO-рейтингов команд |
| `wr_diff_5/10/20` | Разница winrate за 5/10/20 матчей |
| `h2h_wr` | H2H winrate team1 vs team2 |
| `rest_diff` | Разница дней отдыха |
| `n_games` | Формат матча (BO1/BO3/BO5) |

## Value Edge
Ставка рекомендуется только при **Edge > 3%** (настраивается через `--min-edge`).

```
Edge = P_наша × коэффициент − 1
```

## API
PandaScore: https://pandascore.co — есть бесплатный тариф (100 req/min).
