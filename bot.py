#!/usr/bin/env python3
"""
Telegram-бот для прогнозов CS:GO матчей.
Запуск: python bot.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import asyncio

from data.collector import DataCollector
from data.features import build_features
from models.predictor import MatchPredictor
from utils.bankroll import value_edge, kelly_fraction

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GAME = "cs-go"

# Загружаем модель и данные один раз при старте
print("[→] Загружаем данные и модель...")
try:
    df_raw = DataCollector.load_local(game=GAME)
    df = build_features(df_raw)
    predictor = MatchPredictor()
    predictor.load(name=GAME)
    ALL_TEAMS = sorted(set(df["team1_name"].tolist() + df["team2_name"].tolist()))
    print(f"[✓] Готово. Команд в базе: {len(ALL_TEAMS)}")
except Exception as e:
    print(f"[!] Ошибка загрузки: {e}")
    print("[!] Сначала запусти: python main.py fetch --game cs-go && python main.py train --game cs-go")
    sys.exit(1)


class PredictStates(StatesGroup):
    team1 = State()
    team2 = State()
    odds1 = State()
    odds2 = State()


def find_team(query: str) -> list:
    """Поиск команды по частичному совпадению."""
    q = query.lower().strip()
    return [t for t in ALL_TEAMS if q in t.lower()]


def make_prediction(team1: str, team2: str, odds1: float, odds2: float) -> str:
    """Делает прогноз и возвращает текст ответа."""
    t1_rows = df[df["team1_name"].str.contains(team1, case=False, na=False) |
                 df["team2_name"].str.contains(team1, case=False, na=False)]
    t2_rows = df[df["team1_name"].str.contains(team2, case=False, na=False) |
                 df["team2_name"].str.contains(team2, case=False, na=False)]

    if t1_rows.empty:
        return f"❌ Команда *{team1}* не найдена в базе. Напиши /teams чтобы посмотреть список."
    if t2_rows.empty:
        return f"❌ Команда *{team2}* не найдена в базе. Напиши /teams чтобы посмотреть список."

    last1 = t1_rows.iloc[-1]
    last2 = t2_rows.iloc[-1]

    fcols = predictor.fcols
    synth = {}
    for col in fcols:
        if col.endswith("1") or "1_" in col:
            synth[col] = last1.get(col, 0)
        elif col.endswith("2") or "2_" in col:
            synth[col] = last2.get(col, 0)
        else:
            synth[col] = (last1.get(col, 0) + last2.get(col, 0)) / 2

    X = np.nan_to_num(np.array([[synth.get(c, 0) for c in fcols]]), nan=0.0)
    X_sc = predictor.scaler.transform(X)
    prob1 = float(predictor.model.predict_proba(X_sc)[0, 1])
    prob2 = 1 - prob1

    e1 = value_edge(prob1, odds1)
    e2 = value_edge(prob2, odds2)
    k1 = kelly_fraction(prob1, odds1)
    k2 = kelly_fraction(prob2, odds2)

    best_e = max(e1, e2)
    if e1 >= e2:
        best, best_k, best_prob = team1, k1, prob1
    else:
        best, best_k, best_prob = team2, k2, prob2

    bar1 = "█" * int(prob1 * 20)
    bar2 = "█" * int(prob2 * 20)

    text = (
        f"⚔️ *{team1}* vs *{team2}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Вероятности:*\n"
        f"{team1}: {prob1:.1%} `{bar1}`\n"
        f"{team2}: {prob2:.1%} `{bar2}`\n\n"
        f"💰 *Коэффициенты:*\n"
        f"{team1}: {odds1} → Edge: {e1:+.2%}\n"
        f"{team2}: {odds2} → Edge: {e2:+.2%}\n\n"
    )

    if best_e > 0:
        text += (
            f"✅ *СТАВИТЬ НА: {best}*\n"
            f"Kelly: {best_k:.2%} от банкролла\n"
            f"Например: при 10 000₽ → ставь {10000 * best_k:.0f}₽"
        )
    else:
        text += "❌ *Нет value — ставка не рекомендована*"

    return text


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "👋 Привет! Я прогнозирую исходы CS:GO матчей.\n\n"
        "Команды:\n"
        "/predict — сделать прогноз\n"
        "/teams — список команд в базе\n"
        "/help — помощь",
        parse_mode="Markdown"
    )


@dp.message(Command("help"))
async def help_cmd(message: Message):
    await message.answer(
        "📖 *Как пользоваться:*\n\n"
        "1. Напиши /predict\n"
        "2. Введи название первой команды\n"
        "3. Введи название второй команды\n"
        "4. Введи коэффициент на первую команду (напр. 1.85)\n"
        "5. Введи коэффициент на вторую команду (напр. 2.10)\n\n"
        "Получишь прогноз с вероятностями и рекомендацией по ставке.\n\n"
        "Названия команд можно вводить частично — я найду сам.",
        parse_mode="Markdown"
    )


@dp.message(Command("teams"))
async def teams_cmd(message: Message):
    text = "📋 *Команды в базе:*\n\n" + ", ".join(ALL_TEAMS[:80])
    if len(ALL_TEAMS) > 80:
        text += f"\n\n...и ещё {len(ALL_TEAMS) - 80} команд"
    await message.answer(text, parse_mode="Markdown")


@dp.message(Command("predict"))
async def predict_start(message: Message, state: FSMContext):
    await state.set_state(PredictStates.team1)
    await message.answer("🏆 Введи название *первой* команды:", parse_mode="Markdown")


@dp.message(PredictStates.team1)
async def predict_team1(message: Message, state: FSMContext):
    query = message.text.strip()
    found = find_team(query)

    if not found:
        await message.answer(f"❌ Команда '{query}' не найдена. Попробуй ещё раз или /teams")
        return
    if len(found) > 1:
        await message.answer(
            f"🔍 Найдено несколько:\n" + "\n".join(f"• {t}" for t in found[:8]) +
            "\n\nВведи точнее:"
        )
        return

    await state.update_data(team1=found[0])
    await state.set_state(PredictStates.team2)
    await message.answer(f"✅ *{found[0]}*\n\nТеперь введи название *второй* команды:", parse_mode="Markdown")


@dp.message(PredictStates.team2)
async def predict_team2(message: Message, state: FSMContext):
    query = message.text.strip()
    found = find_team(query)

    if not found:
        await message.answer(f"❌ Команда '{query}' не найдена. Попробуй ещё раз или /teams")
        return
    if len(found) > 1:
        await message.answer(
            f"🔍 Найдено несколько:\n" + "\n".join(f"• {t}" for t in found[:8]) +
            "\n\nВведи точнее:"
        )
        return

    await state.update_data(team2=found[0])
    await state.set_state(PredictStates.odds1)
    data = await state.get_data()
    await message.answer(
        f"✅ *{found[0]}*\n\n"
        f"Введи коэффициент букмекера на *{data['team1']}* (напр. 1.85):",
        parse_mode="Markdown"
    )


@dp.message(PredictStates.odds1)
async def predict_odds1(message: Message, state: FSMContext):
    try:
        odds1 = float(message.text.strip().replace(",", "."))
        if odds1 <= 1.0:
            raise ValueError
    except ValueError:
        await message.answer("❌ Некорректный коэффициент. Введи число больше 1 (напр. 1.85):")
        return

    await state.update_data(odds1=odds1)
    await state.set_state(PredictStates.odds2)
    data = await state.get_data()
    await message.answer(
        f"Введи коэффициент на *{data['team2']}*:",
        parse_mode="Markdown"
    )


@dp.message(PredictStates.odds2)
async def predict_odds2(message: Message, state: FSMContext):
    try:
        odds2 = float(message.text.strip().replace(",", "."))
        if odds2 <= 1.0:
            raise ValueError
    except ValueError:
        await message.answer("❌ Некорректный коэффициент. Введи число больше 1 (напр. 2.10):")
        return

    data = await state.get_data()
    await state.clear()

    await message.answer("🔄 Считаю...")

    result = make_prediction(data["team1"], data["team2"], data["odds1"], odds2)
    await message.answer(result, parse_mode="Markdown")
    await message.answer("Ещё прогноз? → /predict")

@dp.message(Command("update"))
async def update_cmd(message: Message):
    await message.answer("🔄 Обновляю данные и переобучаю модель, подожди пару минут...")
    try:
        import subprocess
        subprocess.run(["python", "main.py", "fetch", "--game", "cs-go", "--days", "180"], check=True)
        subprocess.run(["python", "main.py", "train", "--game", "cs-go"], check=True)

        global df, predictor, ALL_TEAMS
        df_raw = DataCollector.load_local(game=GAME)
        df = build_features(df_raw)
        predictor = MatchPredictor()
        predictor.load(name=GAME)
        ALL_TEAMS = sorted(set(df["team1_name"].tolist() + df["team2_name"].tolist()))

        await message.answer("✅ Готово! Данные обновлены, модель переобучена.")
    except Exception as e:
        await message.answer(f"❌ Ошибка: {e}")

async def main():
    if not BOT_TOKEN:
        print("[!] TELEGRAM_BOT_TOKEN не задан в .env")
        sys.exit(1)
    print("[✓] Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
