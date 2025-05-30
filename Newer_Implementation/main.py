import csv
import os
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple, Optional, Dict
import random


def load_and_prepare_games(csv_path: str, stat_key: str) -> List[Dict]:
    games = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stat_val = float(row[stat_key])
                home = row.get("home", "false").lower() == "true"
                date = datetime.strptime(row["date"], "%Y-%m-%d")
            except (ValueError, KeyError):
                continue

            year = date.year
            month = date.month
            season_start = year if month >= 10 else year - 1
            season = f"{season_start}-{str(season_start+1)[-2:]}"

            games.append({
                stat_key: stat_val,
                "home": home,
                "date": date,
                "season": season
            })
    return games


def split_games_by_date(games: List[Dict], test_size: int) -> Tuple[List[Dict], List[Dict]]:
    games.sort(key=lambda x: x["date"])
    return games[:-test_size], games[-test_size:]


def extract_features_and_labels(games: List[Dict], stat_key: str, market_line: float) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    rolling_stats = []
    prev_date = None

    for g in games:
        val = g[stat_key]
        rolling_stats.append(val)
        if len(rolling_stats) > 10:
            rolling_stats.pop(0)
        if len(rolling_stats) < 3:
            continue

        avg_last3 = np.mean(rolling_stats[-3:])
        home_flag = 1 if g["home"] else 0
        base_feature = avg_last3 - market_line
        interaction_home_trend = base_feature * home_flag

        if prev_date is None:
            days_rest = 3
        else:
            days_rest = (g["date"] - prev_date).days
        prev_date = g["date"]

        features = [base_feature, interaction_home_trend, days_rest]
        label = 1 if val >= market_line else 0
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray, model_type: Optional[str] = None, model=None):
    if model is not None:
        model.fit(X, y)
        return model
    if model_type == "logreg":
        model = LogisticRegression()
    elif model_type == "rf":
        model = RandomForestClassifier()
    else:
        raise ValueError("Unsupported model_type or no model provided.")
    model.fit(X, y)
    return model


def calculate_ev(model, games: List[Dict], stat_key: str, market_line: float, threshold: float = 0):
    total_profit, hits, bets = 0, 0, 0
    X, _ = extract_features_and_labels(games, stat_key, market_line)
    for i, g in enumerate(games):
        if i >= len(X):
            break
        features = X[i]
        val = g[stat_key]
        prob = model.predict_proba([features])[0][1]
        ev = prob - (1 - prob)
        if ev > threshold:
            outcome = 1 if val >= market_line else 0
            total_profit += (1 if outcome else -1)
            hits += outcome
            bets += 1
    return {
        "avg_ev": total_profit / bets if bets else 0,
        "win_rate": hits / bets if bets else 0,
        "profit": total_profit,
        "bets": bets
    }


def predict_ev_for_prop(model, games: List[Dict], stat_key: str, market_line: float, recent_n: int = 3) -> Tuple[float, float]:
    stats = [g[stat_key] for g in games]
    homes = [1 if g["home"] else 0 for g in games]
    if len(stats) < recent_n:
        raise ValueError(f"Not enough games ({len(stats)}) to compute recent_{recent_n} features.")
    avg_stat = np.mean(stats[-recent_n:])
    avg_home = int(round(np.mean(homes[-recent_n:])))
    features = [avg_stat - market_line, (avg_stat - market_line) * avg_home, 1 if homes[-1] else 3]
    prob = model.predict_proba([features])[0][1]
    ev = round(2 * prob - 1, 4)
    return prob, ev


def log_prediction_to_global_csv(player: str, stat_key: str, market_line: float, model,
                                 games: List[Dict], model_choice: str, num_games_used: int):
    prob, ev = predict_ev_for_prop(model, games, stat_key, market_line)
    print(f"Predicted Probability% : {prob:.2%}")
    print(f"EV: {ev:.2f}")

    # Always log the prediction, regardless of EV
    log_path = "logs/EV_prop_prediction_log.csv"
    os.makedirs("logs", exist_ok=True)
    header = ["Date", "Player", "Stat", "Line", "Model", "Bet Probability", "EV", "Train Games", "Over/Under"]
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.today().strftime("%Y-%m-%d"),
            player,
            stat_key,
            market_line,
            model_choice,
            round(prob, 4),
            ev,
            num_games_used,
            ""
        ])
    print(f"[✓] Logged to {log_path}")


def main():
    player = input("Enter player name (e.g., amen_thompson): ").strip().lower().replace(" ", "_")
    stat_key = input("Enter stat (e.g., PTS, AST, PRA): ").strip().upper()
    try:
        market_line = float(input("Enter the market line (e.g., 14.5): "))
    except ValueError:
        print("Invalid market line.")
        return

    train_input = input("How many past games to train on? (number or 'all'): ").strip().lower()
    model_choice = input("Choose model (logreg or rf): ").strip().lower()
    if model_choice not in ["logreg", "rf"]:
        print("Unsupported model type.")
        return

    csv_path = f"csv_exports/{player}.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found at: {csv_path}")
        return

    games = load_and_prepare_games(csv_path, stat_key)

    if train_input == "all":
        train_games = games
    else:
        try:
            n = int(train_input)
        except ValueError:
            print("Invalid number for training size.")
            return
        if n > len(games):
            print("Training size exceeds available games.")
            return
        train_games = games[-n:]

    X_train, y_train = extract_features_and_labels(train_games, stat_key, market_line)
    model = train_model(X_train, y_train, model_type=model_choice)

    # Log prediction and include how many games were used for training
    log_prediction_to_global_csv(
        player,
        stat_key,
        market_line,
        model,
        games,
        model_choice,
        len(train_games)
    )


if __name__ == "__main__":
    main()
