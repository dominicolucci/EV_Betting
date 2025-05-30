import csv
import math
import os
import random
import re
import requests
import statistics
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from NBA_Player_Map import player_name_to_id



# Import logistic regression from scikit-learn and numpy for array handling
from sklearn.linear_model import LogisticRegression
import numpy as np

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
    # Basketball Reference suggests not exceeding ~20 requests per minute.
    # Base delay is set to 3 sec; if total requests exceed 20, we increase the delay proportionally.
    base_delay = 3.0
    multiplier = total_requests / 20.0  # e.g. if total_requests == 20, multiplier = 1
    return base_delay * multiplier

def save_games_to_csv(player_name, games, stat_key):
    folder = "csv_exports"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Normalize filename (e.g., "LeBron James" → "LeBron_James.csv")
    filename_safe = player_name.title().replace(" ", "_")
    filename = os.path.join(folder, f"{filename_safe}.csv")

    # Try to load existing games
    existing_games = []
    if os.path.exists(filename):
        with open(filename, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_games.append(row)

    # Merge and deduplicate by date
    combined = {g["date"]: g for g in existing_games}
    for game in games:
        combined[game["date"]] = {
            "date": game["date"],
            "opponent": game["opponent"],
            "home": game["home"],
            "b2b": game["b2b"],
            "result": game["result"],
            "MIN": game["MIN"],
            stat_key: game.get(stat_key, None),
            "EV_LR": game.get("EV_LR", None)
        }

    # Sort by date
    combined_games = sorted(combined.values(), key=lambda g: datetime.strptime(g["date"], "%Y-%m-%d"))

    # Write to CSV
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["date", "opponent", "home", "b2b", "result", "MIN", stat_key, "EV_LR"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for game in combined_games:
            writer.writerow(game)


# ==========================
# Section 1: Arbscanner Functions (Player Logs & EV Calculation)
# ==========================

# Define a list of rotating user agents and additional headers.
user_agents = [
    # Chrome - Desktop
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.4044.129 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.199 Safari/537.36",
    
    # Firefox - Desktop
    "Mozilla/5.0 (Windows NT 10.0; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:56.0) Gecko/20100101 Firefox/56.0",

    # Safari - macOS & iOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",

    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",

    # Chrome - Mobile (Android)
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-G991U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.85 Mobile Safari/537.36",

    # Opera
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/96.0.4693.80",

    # Brave (mimics Chrome)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
]


def get_random_headers():
    return {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",  # Do Not Track
        "Connection": "keep-alive"
    }

def get_game_logs(player_id, season, stat_key, target_opp=None, filter_home=None, filter_b2b=False, delay=3):
    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{season}/"
    
    max_retries = 5
    retry_delay = delay
    response = None

    for attempt in range(max_retries):
        try:
            headers = get_random_headers()
            time.sleep(random.uniform(2, 4) * (delay / 3))  # base randomized delay
            response = requests.get(url, headers=headers)

            if response.status_code == 429:
                print(f"[!] 429 Too Many Requests on attempt {attempt + 1}. Backing off for {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 1.8  # Exponential backoff
                continue
            elif response.status_code != 200:
                raise Exception(f"Unexpected status code {response.status_code}")
            break  # Success, exit retry loop
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

    games = []
    last_game_date = None
    stat_map = {
        "PTS": 31, "AST": 26, "TRB": 25, "REB": 25,
        "STL": 27, "BLK": 28, "TOV": 29, "FG": 10, "FGA": 11,
        "3P": 13, "3PA": 14, "FT": 20, "FTA": 21, "MP": 8
    }

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

        combo_stat = None
        if stat_key == "PRA":
            combo_stat = ["PTS", "AST", "TRB"]
        elif stat_key == "P+A":
            combo_stat = ["PTS", "AST"]
        elif stat_key == "P+R":
            combo_stat = ["PTS", "TRB"]
        elif stat_key == "R+A":
            combo_stat = ["TRB", "AST"]
        elif stat_key == "S+B":
            combo_stat = ["STL", "BLK"]

        is_home = location != "@"
        is_b2b = (last_game_date is not None and (game_date - last_game_date).days == 1)
        last_game_date = game_date

        if target_opp and opponent != target_opp:
            continue
        if filter_home is not None and is_home != filter_home:
            continue
        if filter_b2b and not is_b2b:
            continue

        if combo_stat:
            total = 0
            for s in combo_stat:
                try:
                    val = float(cells[stat_map[s]].get_text(strip=True))
                    total += val
                except ValueError:
                    total = None
                    break
            stat_value = total
        else:
            try:
                stat_value = float(cells[stat_map[stat_key]].get_text(strip=True))
            except ValueError:
                stat_value = None

        games.append({
            "date": game_date_str,
            "opponent": opponent,
            "home": is_home,
            "b2b": is_b2b,
            "result": result,
            "MIN": minutes,
            stat_key: stat_value
        })

    return games

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
# Section 2: Logistic Regression EV Model with Home/Away & Opponent Strength
# ==========================
def train_logistic_regression(games, stat_key, market_line, include_home=False, include_ortg=False, include_drtg=False):
    X = []
    y = []
    for game in games:
        val = game.get(stat_key)
        if val is None:
            continue
        diff = val - market_line
        features = [diff]
        if include_home:
            features.append(1 if game.get("home") else 0)
        if include_ortg:
            ortg = game.get("opponent_ortg")
            if ortg is None:
                continue
            features.append(ortg)
        if include_drtg:
            drtg = game.get("opponent_drtg")
            if drtg is None:
                continue
            features.append(drtg)
        if any(f is None for f in features):
            continue
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
# Section 3: Depthchart Functions (Injury Information)
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
    headers = get_random_headers()
    # Insert a fixed delay (e.g., 3 seconds) for safety
    time.sleep(random.uniform(2, 4))
    response = requests.get(url, headers=headers)
    response.raise_for_status()
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
# Section 4: Combined Program Functionality
# ==========================
def run_player_log_analysis():
    try:
        num_ids = int(input("How many players will you enter? "))
    except ValueError:
        print("Invalid number entered.")
        return

    player_names_input = input("Enter {} player names (comma-separated, e.g., LeBron James, Luka Doncic): ".format(num_ids)).strip()
    player_names = [name.strip().lower() for name in player_names_input.split(",") if name.strip()]
    if len(player_names) != num_ids:
        print("Please enter exactly {} names.".format(num_ids))
        return

    # Convert player names to IDs
    player_ids = []
    for name in player_names:
        if name in player_name_to_id:
            player_ids.append(player_name_to_id[name])
        else:
            print("Player name '{}' not found in player map. Please add it.".format(name))
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
    
    # Compute dynamic delay based on number of requests (players x seasons)
    global_delay = compute_delay(num_ids, season_input)
    print(f"Applying a delay of ~{global_delay:.2f} seconds per request...")

    include_home_input = input("Include home/away factor? (Y/N): ").strip().lower()
    include_home = True if include_home_input == "y" else False

    for player_id in player_ids:
        all_games = []
        for season in seasons:
            try:
                season_games = get_game_logs(
                    player_id, season, stat_key,
                    target_opp=target_opp,
                    filter_home=filter_home,
                    filter_b2b=filter_b2b,
                    delay=global_delay
                )
                print(f"Scraped {len(season_games)} games for {player_id} in {season}")
                all_games.extend(season_games)
            except Exception as e:
                print(f"Failed to scrape {player_id} for {season}: {e}")

        if all_games:
            save_games_to_csv(player_id, all_games, stat_key)
            print(f"Saved {player_id} stats to CSV.\n")

            avg_ev = calculate_ev(all_games, stat_key, market_line)
            print(f"\n--- Results for {player_id} ---")
            print(f"Traditional EV for {stat_key} (Line {market_line}): {avg_ev:.2f}")

            model = train_logistic_regression(all_games, stat_key, market_line,
                                              include_home=include_home,
                                              include_ortg=False,
                                              include_drtg=False)
            if model is None:
                print("Not enough data to train logistic regression model.")
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
                                                include_ortg=False,
                                                include_drtg=False)
                    p_lose = 1 - p_win
                    ev_lr = (p_win * 1) - (p_lose * 1)
                    ev_list_lr.append(ev_lr)
                    g["EV_LR"] = ev_lr

            if ev_list_lr:
                avg_ev_lr = sum(ev_list_lr) / len(ev_list_lr)
                print(f"Average EV using Logistic Regression: {avg_ev_lr:.2f}")
            else:
                print("No valid game data for Logistic Regression EV calculation.")
        else:
            print("No games found.")


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

def run_kelly_betting_simulator():
    try:
        bankroll = float(input("Enter your total bankroll amount (e.g., 1000): ").strip())
    except ValueError:
        print("Invalid bankroll input.")
        return
    
    mode = input("Are you calculating (I)ndependent bets or a (P)arlay? Enter I or P: ").strip().upper()
    
    if mode not in {"I", "P"}:
        print("Invalid mode selection.")
        return
    
    try:
        num_bets = int(input("Enter number of bets or props: ").strip())
    except ValueError:
        print("Invalid number of bets.")
        return
    
    bets = []
    for i in range(num_bets):
        try:
            prob = float(input(f"Enter your model's probability of winning Bet/Prop #{i+1} (e.g., 0.58): ").strip())
            odds = 2.00  # Fixed decimal odds equivalent to -100
            bets.append((prob, odds))
        except ValueError:
            print("Invalid input for bet/prop.")
            return
    
    if mode == "I":
        print("\nRecommended Kelly Bet Sizes for Independent Bets (Assuming -100 Odds for All Bets):")
        for idx, (p, odds) in enumerate(bets, start=1):
            b = odds - 1
            q = 1 - p
            kelly_fraction = (b * p - q) / b if b != 0 else 0
            kelly_fraction = max(0, kelly_fraction)
            full_bet = bankroll * kelly_fraction
            half_bet = full_bet / 2
            print(f"Bet #{idx}: Probability={p:.2f} →")
            print(f"  Full Kelly: {kelly_fraction:.4f} → Bet Amount: ${full_bet:.2f}")
            print(f"  Half Kelly: {kelly_fraction/2:.4f} → Bet Amount: ${half_bet:.2f}")

    elif mode == "P":
        parlay_prob = 1.0
        parlay_odds = 1.0
        for (p, odds) in bets:
            parlay_prob *= p
            parlay_odds *= odds
        
        b = parlay_odds - 1
        q = 1 - parlay_prob
        kelly_fraction = (b * parlay_prob - q) / b if b != 0 else 0
        kelly_fraction = max(0, kelly_fraction)
        full_bet = bankroll * kelly_fraction
        half_bet = full_bet / 2
        
        print("\nParlay Summary (Assuming -100 Odds for Each Prop):")
        print(f"Combined Probability of Winning: {parlay_prob:.4f}")
        print(f"Combined Decimal Odds: {parlay_odds:.2f}")
        print(f"  Full Kelly: {kelly_fraction:.4f} → Bet Amount: ${full_bet:.2f}")
        print(f"  Half Kelly: {kelly_fraction/2:.4f} → Bet Amount: ${half_bet:.2f}")

    print("\nNote: Kelly calculations are assuming -100 odds (even money) for all props.")

def main():
    while True:
        print("\n=== Combined Basketball Analysis Program ===")
        print("Select an option:")
        print("1. Player Game Log Analysis with Logistic Regression EV")
        print("2. Team Ratings & Injuries Report")
        print("3. Combined Report (Player & Team Analysis)")
        print("4. Kelly Criterion Betting Calculator")
        print("5. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            run_player_log_analysis()
        elif choice == "2":
            run_team_analysis()
        elif choice == "3":
            run_combined_report()
        elif choice == "4":
            run_kelly_betting_simulator()
        elif choice == "5":
            print("See ya")
            break
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    main()
