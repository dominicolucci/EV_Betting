import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def pad_day(date_str):
    return re.sub(r'(\w{3}, \w{3} )(\d)(, \d{4})', r'\g<1>0\g<2>\g<3>', date_str)

def is_long_term_out(update_text):
    """
    Checks if the update text contains indicators of a long-term injury.
    """
    long_term_keywords = [
        "out for season",
        "out indefinitely",
        "remainder of the season",
        "expected to miss several weeks",
        "expected to be out multiple weeks",
        "sidelined for extended period",
        "out multiple games",
        "will not return this season",
        "season-ending"
    ]
    update_lower = update_text.lower()
    return any(keyword in update_lower for keyword in long_term_keywords)

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

        if " - " in update_info:
            update = update_info.split(" - ")[0].strip()
        else:
            update = update_info

        try:
            injury_date = datetime.strptime(date_str, "%a, %b %d, %Y")
        except ValueError:
            continue

        # Only include if team matches and either:
        # 1. Injury is within the last 7 days
        # 2. Injury is long-term (based on phrasing in the update)
        if team.lower() == team_name.lower() and (
            seven_days_ago <= injury_date <= today or is_long_term_out(update)
        ):
            matched_injuries.append({
                "player": player,
                "team": team,
                "date": date_str,
                "update": update
            })

    return matched_injuries

def main():
    team_input = input("Enter the full team name (e.g., 'Denver Nuggets'): ").strip()

    try:
        injuries = get_injuries_for_team(team_input)
        if not injuries:
            print(f"No relevant injuries found for '{team_input}'.")
        else:
            print(f"\nInjuries for '{team_input}':\n")
            for inj in injuries:
                print(f"Player: {inj['player']}")
                print(f"Date:   {inj['date']}")
                print(f"Update: {inj['update']}")
                print("-" * 60)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
