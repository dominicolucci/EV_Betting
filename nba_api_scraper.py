import math
import re
import requests
import time
import random
import os
import pickle
import statistics
from datetime import datetime, timedelta

# Import logistic regression from scikit-learn and numpy for array handling
from sklearn.linear_model import LogisticRegression
import numpy as np

# NEW IMPORTS for nba_api and pandas
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players
import pandas as pd

# ==========================
# Helper Function: Dynamic Delay Calculation
# ==========================
def compute_delay(num_players, season_input):
    total_seasons = 0
    for part in season_input.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            total_seasons += (end - start + 1)
        else:
            total_seasons += 1
    total_requests = num_players * total_seasons
    base_delay = 3.0
    multiplier = total_requests / 20.0  # e.g. if total_requests == 20, multiplier = 1
    return base_delay * multiplier

# ==========================
# Helper Function: Caching Utilities (Retained for team ratings or other uses)
# ==========================
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(player_id, season):
    return os.path.join(CACHE_DIR, f"{player_id}_{season}.pkl")

def load_cache(player_id, season):
    filename = get_cache_filename(player_id, season)
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache for {player_id} {season}: {e}")
    return None

def save_cache(player_id, season, data):
    filename = get_cache_filename(player_id, season)
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving cache for {player_id} {season}: {e}")

# ==========================
# NEW: Function to Fetch Game Logs via nba_api
# ==========================
def find_player_id_by_name(player_name):
    # nba_api expects the full name. This function returns the first matching player.
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        raise ValueError(f"Player '{player_name}' not found.")
    return player_dict[0]['id']

def get_game_logs_nba_api(player_name, season, stat_key, target_opp=None, filter_home=None, filter_b2b=False, delay=3):
    player_id = find_player_id_by_name(player_name)
    
    time.sleep(delay)  # Respectful delay between requests
    game_log = PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
    df = game_log.get_data_frames()[0]
    
    # Ensure date field is datetime and sort chronologically
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE')
    
    games = []
    prev_date = None
    for _, row in df.iterrows():
        # 'MATCHUP' field looks like "vs. CHA" or "@ NYK"
        matchup_parts = row['MATCHUP'].split()
        opp = matchup_parts[-1]
        is_home = "vs." in row['MATCHUP']
        is_b2b = False
        if prev_date is not None:
            if (row['GAME_DATE'] - prev_date).days == 1:
                is_b2b = True
        prev_date = row['GAME_DATE']
        
        if target_opp and opp != target_opp:
            continue
        if filter_home is not None and is_home != filter_home:
            continue
        if filter_b2b and not is_b2b:
            continue

        # Determine the stat value; if it's a combo stat, compute accordingly.
        value = None
        if stat_key in df.columns:
            value = row[stat_key]
        else:
            # Handle some custom combo stat keys
            if stat_key == "PRA":
                value = row['PTS'] + row['REB'] + row['AST']
            elif stat_key == "P+A":
                value = row['PTS'] + row['AST']
            elif stat_key == "P+R":
                value = row['PTS'] + row['REB']
            elif stat_key == "R+A":
                value = row['REB'] + row['AST']
            elif stat_key == "S+B":
                value = row['STL'] + row['BLK']
        
        games.append({
            "date": row['GAME_DATE'].strftime("%Y-%m-%d"),
            "opponent": opp,
            "home": is_home,
            "b2b": is_b2b,
            "result": row['WL'],
            "MIN": row['MIN'],
            stat_key: value
        })
    
    return games

# ==========================
# Original Functions: Calculate EV, Parse Seasons, etc.
# ==========================
def calculate_ev(games, stat_key, market_line, profit_if_win=1, loss_if_lose=1, k=0.5):
    ev_list = []
    for game in games:
        if game[stat_key] is not None:
            diff = game[stat_key] - market_line
            p_win = 1 / (1 + math.exp(-k * diff))
            p_lose = 1 - p_win
            ev = (p_win * profit_if_win) - (p_lose * loss_if_lose)
            ev_list.append(ev)
    return sum(ev_list) / len(ev_list) if ev_list else None

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

# ==========================
# Original Logistic Regression EV Model Functions (Unchanged)
# ==========================
def train_logistic_regression(games, stat_key, market_line, include_home=False, include_ortg=False, include_drtg=False):
    X = []
    y = []
    for game in games:
        val = game[stat_key]
        if val is not None:
            diff = val - market_line
            features = [diff]
            if include_home:
                features.append(1 if game["home"] else 0)
            if include_ortg:
                features.append(game.get("opponent_ortg", 0))
            if include_drtg:
                features.append(game.get("opponent_drtg", 0))
            X.append(features)
            outcome = 1 if val >= market_line else 0
            y.append(outcome)
    if len(X) == 0:
        return None
    model = LogisticRegression()
    model.fit(np.array(X), np.array(y))
    return model

def predict_probability(model, stat_value, market_line, is_home=False, opponent_ortg=None, opponent_drtg=None,
                        include_home=False, include_ortg=False, include_drtg=False):
    diff = stat_value - market_line
    features = [diff]
    if include_home:
        features.append(1 if is_home else 0)
    if include_ortg:
        features.append(opponent_ortg if opponent_ortg is not None else 0)
    if include_drtg:
        features.append(opponent_drtg if opponent_drtg is not None else 0)
    prob = model.predict_proba([features])[0][1]
    return prob

# ==========================
# Section 3: Depthchart Functions (Injury Information) -- Unchanged
# ==========================
def pad_day(date_str):
    return re.sub(r'(\w{3}, \w{3} )(\d)(, \d{4})', r'\g<1>0\g<2>\g<3>', date_str)

def is_long_term_out(update_text):
    long_term_keywords = [
        "out for season", "out indefinitely", "remainder of the season",
        "expected to miss several weeks", "expected to be out multiple weeks",
        "sidelined for extended period", "out multiple games",
        "will not return this season", "season-ending"
    ]
    return any(keyword in update_text.lower() for keyword in long_term_keywords)

def get_injuries_for_team(team_name):
    url = "https://www.basketball-reference.com/friv/injuries.cgi"
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2)",
            "Mozilla/5.0 (X11; Linux x86_64)"
        ]),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive"
    }
    time.sleep(random.uniform(2, 4))
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    injuries_table = soup.find("table", {"id": "injuries"})
    if not injuries_table:
        raise ValueError("Could not find injuries table on the page.")
    matched_injuries = []
    today = datetime.now()
    seven_days_ago = today - timedelta(days=7)
    tbody = injuries_table.find("tbody")
    rows = tbody.find_all("tr") if tbody else []
    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) < 4:
            continue
        player = cells[0].text.strip()
        team = cells[1].text.strip()
        date_str = pad_day(cells[2].text.strip())
        update_info = cells[3].text.strip()
        update = update_info.split(" - ")[0].strip() if " - " in update_info else update_info
        try:
            injury_date = datetime.strptime(date_str, "%a, %b %d, %Y")
        except ValueError:
            continue
        if team.lower() == team_name.lower() and (seven_days_ago <= injury_date <= today or is_long_term_out(update)):
            matched_injuries.append({
                "player": player,
                "team": team,
                "date": date_str,
                "update": update
            })
    return matched_injuries

# ==========================
# Section 4: Teamstats Functions (Team Ratings) -- Unchanged
# ==========================
def get_team_ratings(team_abbreviation):
    url = "https://www.basketball-reference.com/leagues/NBA_2025_ratings.html"
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2)",
            "Mozilla/5.0 (X11; Linux x86_64)"
        ]),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive"
    }
    time.sleep(random.uniform(2, 4))
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to load ratings page: {response.status_code}")
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "ratings"})
    if not table:
        raise Exception("Ratings table not found.")
    team_mapping = {
        "OKC": "Oklahoma City Thunder", "CLE": "Cleveland Cavaliers",
        "BOS": "Boston Celtics", "HOU": "Houston Rockets",
        "MIN": "Minnesota Timberwolves", "MEM": "Memphis Grizzlies",
        "LAC": "Los Angeles Clippers", "DEN": "Denver Nuggets",
        "NYK": "New York Knicks", "GSW": "Golden State Warriors",
        "DET": "Detroit Pistons", "MIL": "Milwaukee Bucks",
        "IND": "Indiana Pacers", "LAL": "Los Angeles Lakers",
        "DAL": "Dallas Mavericks", "SAC": "Sacramento Kings",
        "MIA": "Miami Heat", "ORL": "Orlando Magic",
        "ATL": "Atlanta Hawks", "PHX": "Phoenix Suns",
        "CHI": "Chicago Bulls", "SAS": "San Antonio Spurs",
        "POR": "Portland Trail Blazers", "TOR": "Toronto Raptors",
        "PHI": "Philadelphia 76ers", "BKN": "Brooklyn Nets",
        "NOP": "New Orleans Pelicans", "UTA": "Utah Jazz",
        "CHA": "Charlotte Hornets", "WAS": "Washington Wizards"
    }
    target_team = team_mapping.get(team_abbreviation.upper())
    if not target_team:
        raise ValueError(f"Unknown team abbreviation: {team_abbreviation}")
    for row in table.tbody.find_all("tr"):
        team_cell = row.find("td", {"data-stat": "team_name"})
        if team_cell and team_cell.text.strip() == target_team:
            ortg = float(row.find("td", {"data-stat": "off_rtg"}).text)
            drtg = float(row.find("td", {"data-stat": "def_rtg"}).text)
            nrtg = float(row.find("td", {"data-stat": "net_rtg"}).text)
            return {"Team": team_abbreviation.upper(), "ORtg": ortg, "DRtg": drtg, "NRtg": nrtg}
    raise Exception(f"Team {team_abbreviation} not found in ratings table.")

# ==========================
# Section 5: Combined Program Functionality
# ==========================
def run_player_log_analysis():
    try:
        num_ids = int(input("How many players will you enter? "))
    except ValueError:
        print("Invalid number entered.")
        return

    player_names_input = input("Enter {} player names (comma-separated, e.g., LeBron James, Luka Doncic): ".format(num_ids)).strip()
    player_names = [name.strip() for name in player_names_input.split(",") if name.strip()]
    if len(player_names) != num_ids:
        print("Please enter exactly {} names.".format(num_ids))
        return

    stat_key = input("Enter stat (e.g., PTS, AST, PRA, P+A, P+R, R+A, S+B): ").strip().upper()
    try:
        market_line = float(input("Enter the market line (e.g., 25.5): ").strip())
    except ValueError:
        print("Invalid market line. Please enter a number.")
        return

    target_opp = input("Enter opponent abbreviation (e.g., CHI), or leave blank for all: ").strip().upper() or None
    home_filter = input("Only home games? (Y/N or blank for both): ").strip().lower()
    filter_home = True if home_filter == "y" else False if home_filter == "n" else None
    b2b_filter = input("Only back-to-back games? (Y/N or blank for both): ").strip().lower()
    filter_b2b = (b2b_filter == "y")
    season_input = input("Enter seasons (e.g., 2022,2023 or 2021-2025): ").strip()
    seasons = parse_seasons(season_input)
    
    global_delay = compute_delay(num_ids, season_input)
    print(f"Applying a delay of ~{global_delay:.2f} seconds per request...")

    include_home_input = input("Include home/away factor? (Y/N): ").strip().lower()
    include_home = True if include_home_input == "y" else False
    include_ortg_input = input("Include opponent offensive rating (ORTG)? (Y/N): ").strip().lower()
    include_ortg = True if include_ortg_input == "y" else False
    include_drtg_input = input("Include opponent defensive rating (DRTG)? (Y/N): ").strip().lower()
    include_drtg = True if include_drtg_input == "y" else False

    for player_name in player_names:
        all_games = []
        for season in seasons:
            try:
                season_games = get_game_logs_nba_api(
                    player_name, season, stat_key,
                    target_opp=target_opp,
                    filter_home=filter_home,
                    filter_b2b=filter_b2b,
                    delay=global_delay
                )
                print("Fetched {} games for {} in {}".format(len(season_games), player_name, season))
                all_games.extend(season_games)
            except Exception as e:
                print("Failed to fetch {} for {}: {}".format(player_name, season, e))
        if not all_games:
            print("No games found for {}.".format(player_name))
            continue

        # For each game, add opponent ORTG and DRTG using get_team_ratings.
        for game in all_games:
            try:
                opp_stats = get_team_ratings(game["opponent"])
                game["opponent_ortg"] = opp_stats["ORtg"]
                game["opponent_drtg"] = opp_stats["DRtg"]
            except Exception as e:
                game["opponent_ortg"] = None
                game["opponent_drtg"] = None

        avg_ev = calculate_ev(all_games, stat_key, market_line)
        print("\n--- Results for {} ---".format(player_name))
        print("Traditional EV for {} (Line {}): {:.2f}".format(stat_key, market_line, avg_ev))
        model = train_logistic_regression(all_games, stat_key, market_line,
                                          include_home=include_home,
                                          include_ortg=include_ortg,
                                          include_drtg=include_drtg)
        if model is None:
            print("Not enough data to train logistic regression model for {}.".format(player_name))
            continue

        ev_list_lr = []
        for g in all_games:
            val = g[stat_key]
            if val is not None:
                p_win = predict_probability(model, val, market_line,
                                            is_home=g["home"],
                                            opponent_ortg=g.get("opponent_ortg"),
                                            opponent_drtg=g.get("opponent_drtg"),
                                            include_home=include_home,
                                            include_ortg=include_ortg,
                                            include_drtg=include_drtg)
                p_lose = 1 - p_win
                ev_lr = (p_win * 1) - (p_lose * 1)
                ev_list_lr.append(ev_lr)
                print("Date: {} | Opp: {} | {}: {} | Home: {} | ORTG: {} | DRTG: {} | EV (LR): {:.2f}".format(
                    g['date'], g['opponent'], stat_key, g[stat_key], g["home"],
                    g.get("opponent_ortg", "N/A"),
                    g.get("opponent_drtg", "N/A"),
                    ev_lr))
        if ev_list_lr:
            avg_ev_lr = sum(ev_list_lr) / len(ev_list_lr)
            print("Average EV using Logistic Regression: {:.2f}".format(avg_ev_lr))
        else:
            print("No valid game data for Logistic Regression EV calculation for {}.".format(player_name))

def run_team_analysis():
    player_team_abbrev = input("Enter abbreviation for player's team (e.g., LAC, BOS): ").strip().upper()
    opponent_team_abbrev = input("Enter abbreviation for opponent's team (e.g., DEN, GSW): ").strip().upper()
    try:
        player_team_stats = get_team_ratings(player_team_abbrev)
        opponent_team_stats = get_team_ratings(opponent_team_abbrev)
        print("\n{} Team Ratings:".format(player_team_stats['Team']))
        print("  Offensive Rating (ORtg): {}".format(player_team_stats['ORtg']))
        print("  Defensive Rating (DRtg): {}".format(player_team_stats['DRtg']))
        print("  Net Rating (NRtg):       {}".format(player_team_stats['NRtg']))
        print("\n{} Opponent Ratings:".format(opponent_team_stats['Team']))
        print("  Offensive Rating (ORtg): {}".format(opponent_team_stats['ORtg']))
        print("  Defensive Rating (DRtg): {}".format(opponent_team_stats['DRtg']))
        print("  Net Rating (NRtg):       {}".format(opponent_team_stats['NRtg']))
    except Exception as e:
        print("Error retrieving team ratings: {}".format(e))
        return

    team_mapping = {
        "OKC": "Oklahoma City Thunder", "CLE": "Cleveland Cavaliers",
        "BOS": "Boston Celtics", "HOU": "Houston Rockets",
        "MIN": "Minnesota Timberwolves", "MEM": "Memphis Grizzlies",
        "LAC": "Los Angeles Clippers", "DEN": "Denver Nuggets",
        "NYK": "New York Knicks", "GSW": "Golden State Warriors",
        "DET": "Detroit Pistons", "MIL": "Milwaukee Bucks",
        "IND": "Indiana Pacers", "LAL": "Los Angeles Lakers",
        "DAL": "Dallas Mavericks", "SAC": "Sacramento Kings",
        "MIA": "Miami Heat", "ORL": "Orlando Magic",
        "ATL": "Atlanta Hawks", "PHX": "Phoenix Suns",
        "CHI": "Chicago Bulls", "SAS": "San Antonio Spurs",
        "POR": "Portland Trail Blazers", "TOR": "Toronto Raptors",
        "PHI": "Philadelphia 76ers", "BKN": "Brooklyn Nets",
        "NOP": "New Orleans Pelicans", "UTA": "Utah Jazz",
        "CHA": "Charlotte Hornets", "WAS": "Washington Wizards"
    }
    player_team_full = team_mapping.get(player_team_abbrev)
    opponent_team_full = team_mapping.get(opponent_team_abbrev)
    if not player_team_full or not opponent_team_full:
        print("Could not determine full team names for injury reports.")
        return

    try:
        player_injuries = get_injuries_for_team(player_team_full)
        opponent_injuries = get_injuries_for_team(opponent_team_full)
        print("\nInjury Report for {}:".format(player_team_full))
        if player_injuries:
            for inj in player_injuries:
                print("  Player: {} | Date: {} | Update: {}".format(inj['player'], inj['date'], inj['update']))
        else:
            print("  No recent or long-term injuries found.")
        print("\nInjury Report for {}:".format(opponent_team_full))
        if opponent_injuries:
            for inj in opponent_injuries:
                print("  Player: {} | Date: {} | Update: {}".format(inj['player'], inj['date'], inj['update']))
        else:
            print("  No recent or long-term injuries found.")
    except Exception as e:
        print("Error retrieving injuries: {}".format(e))

def run_combined_report():
    print("\n--- Player Game Log & EV Analysis ---")
    run_player_log_analysis()
    print("\n--- Team Ratings & Injuries Report ---")
    run_team_analysis()

def main():
    while True:
        print("\n=== Combined Basketball Analysis Program ===")
        print("Select an option:")
        print("1. Player Game Log Analysis with Logistic Regression EV")
        print("2. Team Ratings & Injuries Report")
        print("3. Combined Report (Player & Team Analysis)")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            run_player_log_analysis()
        elif choice == "2":
            run_team_analysis()
        elif choice == "3":
            run_combined_report()
        elif choice == "4":
            print("See ya")
            break
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    main()
