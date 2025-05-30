import os
import csv
from datetime import datetime
from model_util import load_and_prepare_games, extract_features_and_labels, train_model

LOG_FILE = "logs/ev_hit_log.csv"

def initialize_csv():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "player", "stat", "line", "model_prob", "ev", "result"])

def log_model_prediction():
    date = datetime.today().strftime("%Y-%m-%d")
    player = input("Player name (e.g., cade_cunningham): ").strip().lower().replace(" ", "_")
    stat = input("Stat (PTS, AST, etc.): ").strip().upper()
    line = float(input("Market line (e.g., 17.5): "))

    csv_path = f"csv_exports/{player}.csv"
    if not os.path.exists(csv_path):
        print(f"[!] Missing CSV for {player}")
        return

    games = load_and_prepare_games(csv_path, stat)
    if len(games) < 10:
        print("Not enough data to train model.")
        return

    X, y = extract_features_and_labels(games, stat, line)
    model = train_model(X, y, model_type="logreg")

    # Use latest game as simulation for today's inputs
    latest = games[-1]
    features = [latest[stat] - line, 1 if latest["home"] else 0]
    prob = model.predict_proba([features])[0][1]
    ev = round(2 * prob - 1, 4)

    row = [date, player, stat, line, round(prob, 4), ev, ""]
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f"[✓] Logged: {player} {stat} > {line}, prob={prob:.2f}, EV={ev:.2f}")

def main():
    initialize_csv()
    while True:
        log_model_prediction()
        again = input("Log another? (y/n): ").strip().lower()
        if again != "y":
            break

if __name__ == "__main__":
    main()
