import requests
from bs4 import BeautifulSoup

def get_team_ratings(team_abbreviation):
    """
    Scrapes ORtg, DRtg, and NRtg from the NBA 2025 Team Ratings page for the given team abbreviation.
    """
    url = "https://www.basketball-reference.com/leagues/NBA_2025_ratings.html"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to load ratings page: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "ratings"})

    if not table:
        raise Exception("Ratings table not found.")

    abbrev_map = {
        "OKC": "Oklahoma City Thunder",
        "CLE": "Cleveland Cavaliers",
        "BOS": "Boston Celtics",
        "HOU": "Houston Rockets",
        "MIN": "Minnesota Timberwolves",
        "MEM": "Memphis Grizzlies",
        "LAC": "Los Angeles Clippers",
        "DEN": "Denver Nuggets",
        "NYK": "New York Knicks",
        "GSW": "Golden State Warriors",
        "DET": "Detroit Pistons",
        "MIL": "Milwaukee Bucks",
        "IND": "Indiana Pacers",
        "LAL": "Los Angeles Lakers",
        "DAL": "Dallas Mavericks",
        "SAC": "Sacramento Kings",
        "MIA": "Miami Heat",
        "ORL": "Orlando Magic",
        "ATL": "Atlanta Hawks",
        "PHX": "Phoenix Suns",
        "CHI": "Chicago Bulls",
        "SAS": "San Antonio Spurs",
        "POR": "Portland Trail Blazers",
        "TOR": "Toronto Raptors",
        "PHI": "Philadelphia 76ers",
        "BKN": "Brooklyn Nets",
        "NOP": "New Orleans Pelicans",
        "UTA": "Utah Jazz",
        "CHA": "Charlotte Hornets",
        "WAS": "Washington Wizards"
    }

    target_team = abbrev_map.get(team_abbreviation.upper())
    if not target_team:
        raise ValueError(f"Unknown team abbreviation: {team_abbreviation}")

    for row in table.tbody.find_all("tr"):
        team_cell = row.find("td", {"data-stat": "team_name"})
        if team_cell and team_cell.text.strip() == target_team:
            ortg = float(row.find("td", {"data-stat": "off_rtg"}).text)
            drtg = float(row.find("td", {"data-stat": "def_rtg"}).text)
            nrtg = float(row.find("td", {"data-stat": "net_rtg"}).text)
            return {
                "Team": team_abbreviation.upper(),
                "ORtg": ortg,
                "DRtg": drtg,
                "NRtg": nrtg
            }

    raise Exception(f"Team {team_abbreviation} not found in ratings table.")

def main():
    player_team = input("Enter abbreviation for player's team (e.g., LAC, BOS): ").strip().upper()
    opponent_team = input("Enter abbreviation for opponent's team (e.g., DEN, GSW): ").strip().upper()

    try:
        player_stats = get_team_ratings(player_team)
        opponent_stats = get_team_ratings(opponent_team)

        print(f"\n{player_stats['Team']} Team Ratings:")
        print(f"  Offensive Rating (ORtg): {player_stats['ORtg']}")
        print(f"  Defensive Rating (DRtg): {player_stats['DRtg']}")
        print(f"  Net Rating (NRtg):       {player_stats['NRtg']}")

        print(f"\n{opponent_stats['Team']} Opponent Ratings:")
        print(f"  Offensive Rating (ORtg): {opponent_stats['ORtg']}")
        print(f"  Defensive Rating (DRtg): {opponent_stats['DRtg']}")
        print(f"  Net Rating (NRtg):       {opponent_stats['NRtg']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
