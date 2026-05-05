import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(exist_ok=True)
PANDASCORE_BASE = "https://api.pandascore.co"

class DataCollector:
    def __init__(self, api_token=None):
        self.token = api_token
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})

    def _get(self, endpoint, params=None, max_pages=30):
        if not self.token:
            raise ValueError("API токен не задан")
        url = f"{PANDASCORE_BASE}{endpoint}"
        params = params or {}
        params["per_page"] = 100
        all_results = []
        for page in range(1, max_pages + 1):
            params["page"] = page
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code >= 500:
                    print(f"[!] Сервер {resp.status_code} на стр.{page}, стоп")
                    break
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"[!] Ошибка на стр.{page}: {e}")
                break
            if not data:
                break
            all_results.extend(data)
            print(f"  Страниц: {page}, матчей: {len(all_results)}", end="\r")
            if len(data) < 100:
                break
            time.sleep(0.3)
        print()
        return all_results

    def fetch_matches(self, game="csgo", days_back=180):
        since_dt = datetime.utcnow() - timedelta(days=days_back)
        print(f"[→] Загружаем матчи CS:GO/CS2 за {days_back} дней...")
        # Правильный endpoint по документации PandaScore
        raw = self._get("/csgo/matches/past")
        records = []
        for m in raw:
            date = m.get("end_at") or m.get("begin_at")
            if not date:
                continue
            try:
                match_dt = datetime.fromisoformat(date.replace("Z", "+00:00")).replace(tzinfo=None)
                if match_dt < since_dt:
                    continue
            except Exception:
                continue
            winner_id = (m.get("winner") or {}).get("id")
            if not winner_id:
                continue
            opponents = m.get("opponents") or []
            if len(opponents) < 2:
                continue
            teams = {}
            for o in opponents:
                opp = o.get("opponent") or {}
                if opp.get("id") and opp.get("name"):
                    teams[opp["id"]] = opp["name"]
            if len(teams) < 2:
                continue
            ids = list(teams.keys())
            records.append({
                "match_id": m["id"], "game": "cs-go", "date": date,
                "tournament": (m.get("tournament") or {}).get("name", ""),
                "series": (m.get("serie") or {}).get("full_name", ""),
                "team1_id": ids[0], "team1_name": teams[ids[0]],
                "team2_id": ids[1], "team2_name": teams[ids[1]],
                "winner_id": winner_id,
                "label": 1 if winner_id == ids[0] else 0,
                "n_games": m.get("number_of_games") or 1,
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            path = DATA_DIR / "cs-go_matches.csv"
            df.to_csv(path, index=False)
            print(f"[✓] Сохранено {len(df)} матчей → {path}")
        else:
            print("[!] Матчей не найдено")
        return df

    @staticmethod
    def load_local(game="cs-go"):
        path = DATA_DIR / f"{game}_matches.csv"
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        return df.reset_index(drop=True)
