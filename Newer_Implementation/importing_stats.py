import os
import csv
import time
import random
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from NBA_Player_Map import player_name_to_id
import numpy

# Configurable stat fields to extract
target_stats = ["PTS", "AST", "REB", "TRB", "STL", "BLK", "TOV", "FG", "FGA", "3P", "3PA", "FT", "FTA", "MP"]

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.4044.129 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3; rv:122.0) Gecko/20100101 Firefox/122.0"
]

def get_random_headers():
    return {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive"
    }

def parse_seasons(season_input):
    seasons = []
    for part in season_input.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            seasons.extend(str(year) for year in range(start, end + 1))
        else:
            seasons.append(part)
    return seasons

def get_game_logs(player_id, season, delay=3):
    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{season}/"

    max_retries = 5
    retry_delay = delay
    response = None

    for attempt in range(max_retries):
        try:
            headers = get_random_headers()
            time.sleep(random.uniform(2, 4) * (delay / 3))
            response = requests.get(url, headers=headers)

            if response.status_code == 429:
                print(f"[!] 429 Too Many Requests on attempt {attempt + 1}. Backing off for {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 1.8
                continue
            elif response.status_code != 200:
                raise Exception(f"Unexpected status code {response.status_code}")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to retrieve logs for {player_id} in {season} after {max_retries} attempts.") from e
            print(f"[!] Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay:.1f} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 1.5

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "player_game_log_reg"})
    if not table:
        raise ValueError(f"Game log table not found for season {season}")

    tbody = table.find("tbody")
    if not tbody:
        raise ValueError(f"No table body found for season {season}")

    stat_map = {
        "PTS": 31, "AST": 26, "TRB": 25, "REB": 25,
        "STL": 27, "BLK": 28, "TOV": 29, "FG": 10, "FGA": 11,
        "3P": 13, "3PA": 14, "FT": 20, "FTA": 21, "MP": 8
    }

    games = []
    last_game_date = None

    for row in tbody.find_all("tr"):
        if "thead" in row.get("class", []) or "partial_table" in row.get("class", []):
            continue
        cells = row.find_all(["th", "td"])
        if len(cells) < 34:
            continue

        game_date_str = cells[3].get_text(strip=True)
        location = cells[5].get_text(strip=True)
        opponent = cells[6].get_text(strip=True)
        result = cells[7].get_text(strip=True)
        minutes = cells[9].get_text(strip=True)

        try:
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d")
        except ValueError:
            continue

        game = {
            "date": game_date_str,
            "opponent": opponent,
            "home": location != "@",
            "b2b": (last_game_date is not None and (game_date - last_game_date).days == 1),
            "result": result,
            "MIN": minutes
        }

        last_game_date = game_date

        for stat in target_stats:
            try:
                game[stat] = float(cells[stat_map[stat]].get_text(strip=True))
            except (ValueError, IndexError):
                game[stat] = None

        games.append(game)

    return games

def save_games_to_csv(player_name, new_games):
    import pandas as pd

    folder = "csv_exports"
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = os.path.join(folder, f"{player_name.replace(' ', '_')}.csv")

    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
    else:
        df_existing = pd.DataFrame()

    df_new = pd.DataFrame(new_games)
    if not df_existing.empty:
        df_combined = pd.merge(df_existing, df_new, on="date", how="outer", suffixes=("", "_new"))
        for col in df_new.columns:
            if col != "date" and col + "_new" in df_combined.columns:
                df_combined[col] = df_combined[col].combine_first(df_combined[col + "_new"])
                df_combined.drop(columns=[col + "_new"], inplace=True)
    else:
        df_combined = df_new

    df_combined = df_combined.sort_values("date")
    df_combined.to_csv(filename, index=False)

# ==========================
# Run Program (Standalone Scraper)
# ==========================
if __name__ == "__main__":
    player_input = input("Enter player name (e.g., LeBron James): ").strip().lower()
    if player_input not in player_name_to_id:
        print("Player not found in player map.")
        exit(1)

    season_input = input("Enter seasons to scrape (e.g., 2020,2021-2023 or 'all'): ").strip().lower()

    if season_input == "all":
        first_year = input("Enter player's first season (e.g., 2012): ").strip()
        if not first_year.isdigit():
            print("Invalid input. Please enter a numeric year.")
            exit(1)
        current_year = datetime.now().year
        seasons = [str(year) for year in range(int(first_year), current_year + 1)]
    else:
        seasons = parse_seasons(season_input)

    player_id = player_name_to_id[player_input]

    filename = os.path.join("csv_exports", f"{player_input.replace(' ', '_')}.csv")
    existing_dates = set()

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_dates.add(row["date"])

    all_games = []
    for season in seasons:
        try:
            games = get_game_logs(player_id, season)
            season_dates = {g["date"] for g in games}
            unseen_dates = season_dates - existing_dates

            if not unseen_dates:
                print(f"[✓] Skipping {season} — all games already exist.")
                continue

            new_games = [g for g in games if g["date"] in unseen_dates]
            all_games.extend(new_games)
            existing_dates.update(unseen_dates)
            print(f"[✓] Scraped {len(new_games)} new games for {player_input.title()} in {season}")
        except Exception as e:
            print(f"[!] Failed to scrape {season}: {e}")

    if all_games:
        save_games_to_csv(player_input, all_games)
        print(f"[✓] Saved game logs to csv_exports/{player_input.replace(' ', '_')}.csv")
    else:
        print("[!] No games found to save.")
