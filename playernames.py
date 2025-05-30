import requests
from bs4 import BeautifulSoup, Comment

def get_all_nba_player_names():
    url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    players = []

    # Try to locate the table directly first
    table = soup.find("table", {"id": "per_game_stats"})
    if table:
        # Iterate over all rows in the table
        rows = table.find_all("tr")
        for row in rows:
            # Skip header rows (which often have class "thead")
            if row.get("class") and "thead" in row.get("class"):
                continue
            # Basketball Reference often puts player names in a <th> cell
            player_cell = row.find("th", {"data-stat": "player"})
            # Fallback: sometimes it might be in a <td>
            if not player_cell:
                player_cell = row.find("td", {"data-stat": "player"})
            if player_cell:
                name = player_cell.get_text(strip=True)
                if name and name not in players:
                    players.append(name)
    else:
        # If the table isn't found directly, try inside HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if 'id="per_game_stats"' in comment:
                comment_soup = BeautifulSoup(comment, "html.parser")
                table = comment_soup.find("table", {"id": "per_game_stats"})
                if table:
                    rows = table.find_all("tr")
                    for row in rows:
                        if row.get("class") and "thead" in row.get("class"):
                            continue
                        player_cell = row.find("th", {"data-stat": "player"})
                        if not player_cell:
                            player_cell = row.find("td", {"data-stat": "player"})
                        if player_cell:
                            name = player_cell.get_text(strip=True)
                            if name and name not in players:
                                players.append(name)
                if players:  # exit once we've found the table and extracted names
                    break

    return players

def main():
    print("Scraping NBA player names...\n")
    player_names = get_all_nba_player_names()
    if player_names:
        print("All Current NBA Players (from Per Game Stats):\n")
        for name in player_names:
            print(name)
        print(f"\nTotal: {len(player_names)} players found.")
    else:
        print("No players found. Check HTML parsing or if the table is missing.")

if __name__ == "__main__":
    main()
