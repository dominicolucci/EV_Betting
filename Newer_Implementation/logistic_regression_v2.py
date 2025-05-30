import csv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    player_input = input("Enter player name (e.g., cade_cunningham): ").strip().lower().replace(" ", "_")
    stat_key = input("Enter stat (e.g., PTS, PRA): ").strip().upper()
    try:
        market_line = float(input("Enter the market line (e.g., 24.5): ").strip())
    except ValueError:
        print("Invalid number entered for market line.")
        return

    csv_path = f"csv_exports/{player_input}.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file not found for {player_input}. Expected at: {csv_path}")
        return

    games = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stat_val = float(row[stat_key])
                home = row["home"].lower() == "true"
            except (ValueError, KeyError):
                continue
            games.append({stat_key: stat_val, "home": home})

    if not games:
        print("No valid games found.")
        return

    # Train logistic regression model
    X, y = [], []
    for g in games:
        val = g[stat_key]
        diff = val - market_line
        features = [diff, 1 if g["home"] else 0]
        X.append(features)
        y.append(1 if val >= market_line else 0)

    model = LogisticRegression()
    model.fit(np.array(X), np.array(y))

    # Predict EV for each game
    ev_list = []
    for g in games:
        val = g[stat_key]
        features = [val - market_line, 1 if g["home"] else 0]
        prob = model.predict_proba([features])[0][1]
        ev = (prob * 1) - ((1 - prob) * 1)
        ev_list.append(ev)

    avg_ev = sum(ev_list) / len(ev_list)
    print(f"\nAverage Logistic Regression EV for {player_input} on {stat_key} line {market_line}: {avg_ev:.2f}")

if __name__ == "__main__":
    main()
