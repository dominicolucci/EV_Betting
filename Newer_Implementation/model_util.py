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
                home = row["home"].lower() == "true"
                date = datetime.strptime(row["date"], "%Y-%m-%d")
            except (ValueError, KeyError):
                continue

            year = date.year
            month = date.month
            if month >= 10:
                season_start = year
            else:
                season_start = year - 1
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

def split_games_by_season(games: List[Dict], train_seasons: List[str], test_seasons: List[str]) -> Tuple[List[Dict], List[Dict]]:
    train_games = [g for g in games if g["season"] in train_seasons]
    test_games = [g for g in games if g["season"] in test_seasons]
    return train_games, test_games

def extract_features_and_labels(games: List[Dict], stat_key: str, market_line: float) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for g in games:
        val = g[stat_key]
        features = [val - market_line, 1 if g["home"] else 0]
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
    total_profit = 0
    hits = 0
    bets = 0
    for g in games:
        val = g[stat_key]
        features = [val - market_line, 1 if g["home"] else 0]
        prob = model.predict_proba([features])[0][1]
        ev = prob - (1 - prob)
        if ev > threshold:
            outcome = 1 if val >= market_line else 0
            profit = 1 if outcome else -1
            total_profit += profit
            hits += outcome
            bets += 1
    return {
        "avg_ev": total_profit / bets if bets else 0,
        "win_rate": hits / bets if bets else 0,
        "profit": total_profit,
        "bets": bets
    }

def rolling_ev_evaluation(games: List[Dict], stat_key: str, market_line: float,
                          model_type: str="logreg", n_splits: int=5, ev_threshold: float=0):
    X_full, y_full = extract_features_and_labels(games, stat_key, market_line)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    total_profit = 0
    total_hits = 0
    total_bets = 0
    for train_idx, test_idx in tscv.split(X_full):
        X_train, y_train = X_full[train_idx], y_full[train_idx]
        model = train_model(X_train, y_train, model_type=model_type)
        test_games = [games[i] for i in test_idx]
        result = calculate_ev(model, test_games, stat_key, market_line, threshold=ev_threshold)
        total_profit += result["profit"]
        total_hits += result["win_rate"] * result["bets"]
        total_bets += result["bets"]
    return {
        "avg_ev": total_profit / total_bets if total_bets else 0,
        "win_rate": total_hits / total_bets if total_bets else 0,
        "profit": total_profit,
        "bets": total_bets,
        "splits": n_splits
    }

def run_all_split_evaluations(games, stat_key, market_line, model_type="logreg"):
    print("\nChoose split type:")
    print("1. Last N games")
    print("2. Season-based")
    print("3. Rolling walk-forward")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        test_size = int(input("How many games do you want to test on? "))
        train_games, test_games = split_games_by_date(games, test_size)
        X_train, y_train = extract_features_and_labels(train_games, stat_key, market_line)
        model = train_model(X_train, y_train, model_type=model_type)
        results = calculate_ev(model, test_games, stat_key, market_line)
        print("\n--- Fixed Range Split Results ---")
        print(results)

    elif choice == "2":
        train_seasons = input("Enter train seasons (comma-separated, e.g. 2021-22,2022-23): ").split(",")
        test_seasons = input("Enter test seasons (comma-separated): ").split(",")
        train_games, test_games = split_games_by_season(games, [s.strip() for s in train_seasons], [s.strip() for s in test_seasons])
        X_train, y_train = extract_features_and_labels(train_games, stat_key, market_line)
        model = train_model(X_train, y_train, model_type=model_type)
        results = calculate_ev(model, test_games, stat_key, market_line)
        print("\n--- Season-Based Split Results ---")
        print(results)

    elif choice == "3":
        splits = int(input("Number of rolling splits to perform? "))
        results = rolling_ev_evaluation(games, stat_key, market_line, model_type=model_type, n_splits=splits)
        print("\n--- Rolling Walk-Forward Results ---")
        print(results)

    else:
        print("Invalid choice. Please select 1, 2, or 3.")
