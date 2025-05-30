import math
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import json
from NBA_Player_Map import player_name_to_id


# Import logistic regression from scikit-learn and numpy for array handling
from sklearn.linear_model import LogisticRegression
import numpy as np

# ==========================
# Section 1: Arbscanner Functions (Player Logs & EV Calculation)
# ==========================

def get_game_logs(player_id, season, stat_key, target_opp=None, filter_home=None, filter_b2b=False):
    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/{season}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching page for season {season}: Status code {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "player_game_log_reg"})
    if not table:
        raise ValueError(f"Game log table not found for season {season}")
    
    tbody = table.find("tbody")
    if not tbody:
        raise ValueError(f"No table body found for season {season}")

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

        stat_map = {
            "PTS": 31, "AST": 26, "TRB": 25, "REB": 25,
            "STL": 27, "BLK": 28, "TOV": 29, "FG": 10, "FGA": 11,
            "3P": 13, "3PA": 14, "FT": 20, "FTA": 21, "MP": 8
        }

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

# Traditional EV calculation using a fixed logistic function.
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
# Section 2: Logistic Regression EV Model
# ==========================
def train_logistic_regression(games, stat_key, market_line):
    """
    Train a logistic regression model using the games data.
    Features: the difference (stat - market_line)
    Outcome: 1 if player achieved or exceeded the market line, 0 otherwise.
    """
    X = []
    y = []
    for game in games:
        val = game[stat_key]
        if val is not None:
            diff = val - market_line
            X.append([diff])  # Feature as a single-element list
            outcome = 1 if val >= market_line else 0
            y.append(outcome)
    if len(X) == 0:
        return None  # Not enough data to train on
    model = LogisticRegression()
    model.fit(np.array(X), np.array(y))
    return model

def predict_probability(model, stat_value, market_line):
    """
    Use the trained logistic regression model to predict the probability
    of exceeding the market line given a stat value.
    """
    diff = stat_value - market_line
    prob = model.predict_proba([[diff]])[0][1]  # Probability of class 1
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
    headers = {"User-Agent": "Mozilla/5.0"}
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
        if team.lower() == team_name.lower() and (
            seven_days_ago <= injury_date <= today or is_long_term_out(update)
        ):
            matched_injuries.append({
                "player": player, "team": team, "date": date_str, "update": update
            })
    return matched_injuries

# ==========================
# Section 4: Teamstats Functions (Team Ratings)
# ==========================
def get_team_ratings(team_abbreviation):
    url = "https://www.basketball-reference.com/leagues/NBA_2025_ratings.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to load ratings page: {response.status_code}")
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
        "MIA": "Miami Heat", "ORL": "Orlando Magic", "ATL": "Atlanta Hawks",
        "PHX": "Phoenix Suns", "CHI": "Chicago Bulls",
        "SAS": "San Antonio Spurs", "POR": "Portland Trail Blazers",
        "TOR": "Toronto Raptors", "PHI": "Philadelphia 76ers",
        "BKN": "Brooklyn Nets", "NOP": "New Orleans Pelicans",
        "UTA": "Utah Jazz", "CHA": "Charlotte Hornets",
        "WAS": "Washington Wizards"
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
    """Handles player game log analysis and EV calculation using both traditional and logistic regression approaches."""
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

    for player_id in player_ids:
        all_games = []
        for season in seasons:
            try:
                season_games = get_game_logs(
                    player_id, season, stat_key,
                    target_opp=target_opp,
                    filter_home=filter_home,
                    filter_b2b=filter_b2b
                )
                print("Scraped {} games for {} in {}".format(len(season_games), player_id, season))
                all_games.extend(season_games)
            except Exception as e:
                print("Failed to scrape {} for {}: {}".format(player_id, season, e))

        if not all_games:
            print("No games found.")
            continue

        # Traditional EV calculation for comparison
        avg_ev = calculate_ev(all_games, stat_key, market_line)
        print("\n--- Results for {} ---".format(player_id))
        print("Traditional EV for {} (Line {}): {:.2f}".format(stat_key, market_line, avg_ev))

        # Train logistic regression model on historical data
        model = train_logistic_regression(all_games, stat_key, market_line)
        if model is None:
            print("Not enough data to train logistic regression model.")
            continue

        # Use the trained model to predict probabilities and calculate EV per game
        ev_list_lr = []
        for g in all_games:
            val = g[stat_key]
            if val is not None:
                p_win = predict_probability(model, val, market_line)
                p_lose = 1 - p_win
                ev_lr = (p_win * 1) - (p_lose * 1)  # Profit and loss factors are both 1
                ev_list_lr.append(ev_lr)
                print("Date: {} | Opp: {} | {}: {} | EV (LR): {:.2f}".format(
                    g['date'], g['opponent'], stat_key, val, ev_lr))
        if ev_list_lr:
            avg_ev_lr = sum(ev_list_lr) / len(ev_list_lr)
            print("Average EV using Logistic Regression: {:.2f}".format(avg_ev_lr))
        else:
            print("No valid game data for Logistic Regression EV calculation.")

def run_team_analysis():
    """Handles team ratings retrieval and injury reports."""
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
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    main()
