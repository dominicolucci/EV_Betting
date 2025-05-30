import requests
from bs4 import BeautifulSoup
import math
from datetime import datetime

# Player name to Basketball Reference ID map
player_name_to_id = {
    "aaron nesmith": "nesmiaa01",
    "aaron wiggins": "wiggiaa01",
    "aj green": "greenaj01",
    "al horford": "horfoal01",
    "alex len": "lenal01",
    "andrew nembhard": "nembhan01",
    "andrew wiggins": "wiggian01",
    "anthony davis": "davisan02",
    "anthony edwards": "edwaran01",
    "austin reaves": "reaveau01",
    "ben sheppard": "sheppbe01",
    "bismack biyombo": "biyombi01",
    "blake wesley": "weslebl01",
    "bobby portis": "portibo01",
    "brandin podziemski": "podzibr01",
    "brandon ingram": "ingrabr01",
    "brandon miller": "millebr02",
    "bryce mcgowens": "mcgowbr01",
    "cam johnson": "johnsca02",
    "cam reddish": "reddica01",
    "caris levert": "leverca01",
    "cason wallace": "wallaca01",
    "cedi osman": "osmande01",
    "charles bassey": "bassech01",
    "chris duarte": "duartch01",
    "chris livingston": "livinch01",
    "chris paul": "paulch01",
    "christian wood": "woodch01",
    "cj mccollum": "mccolcj01",
    "cody martin": "martico01",
    "cody zeller": "zelleco01",
    "colby jones": "jonesco02",
    "colin castleton": "castlco01",
    "craig porter jr.": "portecr01",
    "damian jones": "jonesda03",
    "damian lillard": "lillada01",
    "daniel gafford": "gaffoda01",
    "danilo gallinari": "gallida01",
    "danuel house jr.": "houseda01",
    "daquan jeffries": "jeffrda01",
    "darius garland": "garlada01",
    "dario saric": "saricda01",
    "davion mitchell": "mitchda01",
    "day'ron sharpe": "sharpda01",
    "de'aaron fox": "foxde01",
    "dean wade": "wadede01",
    "dennis smith jr.": "smithde03",
    "derrick white": "whitede01",
    "drew peterson": "peterdr01",
    "doug mcdermott": "mcderdo01",
    "dorian finney-smith": "finnedo01",
    "dyson daniels": "daniedy01",
    "emoni bates": "batesem01",
    "evan mobley": "mobleev01",
    "frank ntilikina": "ntilifr01",
    "gabe vincent": "vincega01",
    "gary payton ii": "paytoga02",
    "georges niang": "niangge01",
    "giannis antetokounmpo": "antetgi01",
    "gordon hayward": "haywago01",
    "harrison barnes": "barneha02",
    "isaac okoro": "okorois01",
    "isaiah jackson": "jackson01",
    "isaiah joe": "joeis01",
    "ish smith": "smithis01",
    "jabari walker": "walkewa01",
    "jack white": "whiteja02",
    "jacob gilyard": "gilyaja01",
    "jae crowder": "crowdja01",
    "jaime jaquez jr.": "jaqueja01",
    "jalen hood-schifino": "hoodsja01",
    "jalen smith": "smithja04",
    "jalen slawson": "slawsja01",
    "jalen wilson": "wilsoja03",
    "james johnson": "johnsja01",
    "jarrett allen": "allenja01",
    "jarred vanderbilt": "vandeja01",
    "javale mcgee": "mcgeeja01",
    "jaylen brown": "brownja02",
    "jaylin williams": "willija07",
    "jayson tatum": "tatumja01",
    "jeremiah robinson-earl": "robinje01",
    "jeremy sochan": "sochaje01",
    "jett howard": "howarje02",
    "josh giddey": "giddejo01",
    "josh minott": "minotjo01",
    "josh okogie": "okogijo01",
    "jose alvarado": "alvarjo01",
    "jrue holiday": "holidjr01",
    "juan toscano-anderson": "toscaju01",
    "julian champagnie": "champju02",
    "julius randle": "randlju01",
    "kam whitmore": "whitk01",
    "karl-anthony towns": "townska01",
    "keldon johnson": "johnske04",
    "keon johnson": "johnske07",
    "kevin huerter": "huertke01",
    "keyontae johnson": "johnske06",
    "klay thompson": "thompkl01",
    "kristaps porziņģis": "porzikr01",
    "kyle anderson": "anderky01",
    "lamelo ball": "ballla01",
    "larry nance jr.": "nancela02",
    "lebron james": "jamesle01",
    "leaky black": "blackle01",
    "leonard miller": "millele01",
    "lester quinones": "quinole01",
    "lindell wigginton": "wiggili01",
    "lindy waters iii": "waterli01",
    "lonnie walker iv": "walkelo01",
    "luke kornet": "kornelu01",
    "lu dort": "dortlu01",
    "malaki branham": "branhma01",
    "malik beasley": "beaslma01",
    "malik monk": "monkma01",
    "marjon beauchamp": "beaucma01",
    "mark williams": "willima07",
    "matt ryan": "ryanma01",
    "max christie": "chrisma02",
    "max strus": "strusma01",
    "maxwell lewis": "lewisma05",
    "mikal bridges": "bridgmi01",
    "mike conley": "conlemi01",
    "miles bridges": "bridgmi02",
    "monte morris": "morrimo01",
    "moses moody": "moodymo01",
    "naz reid": "reidna01",
    "neemias queta": "quetane01",
    "nick richards": "richani01",
    "nickeil alexander-walker": "alexani01",
    "nic claxton": "claxtni01",
    "naji marshall": "marshna01",
    "noah clowney": "clownno01",
    "obi toppin": "toppiob01",
    "olivier sarr": "sarron01",
    "oshae brissett": "brissos01",
    "oscar tshiebwe": "tshieos01",
    "ousmane dieng": "diengou01",
    "pascal siakam": "siakapa01",
    "patrick beverley": "beverpa01",
    "payton pritchard": "pritcpa01",
    "pj tucker": "tuckepj01",
    "quenton jackson": "jacksqu01",
    "rui hachimura": "hachiru01",
    "sam hauser": "hausesa01",
    "sam merrill": "merrisa01",
    "sandro mamukelashvili": "mamuksa01",
    "sasha vezenkov": "vezensa01",
    "scotty pippen jr.": "pippesc01",
    "shai gilgeous-alexander": "gilgesh01",
    "sidy cissoko": "cissosi01",
    "spencer dinwiddie": "dinwisp01",
    "stephen curry": "curryst01",
    "svi mykhailiuk": "mykhasv01",
    "taurean prince": "princta02",
    "terry rozier": "roziero01",
    "theo maledon": "maledth01",
    "tj mcconnell": "mccontj01",
    "trayce jackson-davis": "jacksja02",
    "tre jones": "jonestr01",
    "tre mann": "manntr01",
    "trey lyles": "lylestr01",
    "trey murphy iii": "murphtr02",
    "tristan thompson": "thomptr01",
    "ty jerome": "jeromty01",
    "tyty washington": "washity02",
    "usman garuba": "garubus01",
    "victor wembanyama": "wembavi01",
    "wendell moore jr.": "moorewe01",
    "zach collins": "colliza01",
    "zion williamson": "willizi01"

    # Add more players as needed
}

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

if __name__ == "__main__":
    num_ids = int(input("How many players will you enter? "))
    player_names_input = input(f"Enter {num_ids} player names (comma-separated, e.g., LeBron James, Luka Doncic): ").strip()
    player_names = [name.strip().lower() for name in player_names_input.split(",") if name.strip()]
    
    if len(player_names) != num_ids:
        print(f"Please enter exactly {num_ids} names.")
        exit()

    player_ids = []
    for name in player_names:
        if name in player_name_to_id:
            player_ids.append(player_name_to_id[name])
        else:
            print(f"Player name '{name}' not found in player map. Please add it.")
            exit()

    stat_key = input("Enter stat (e.g., PTS, AST, PRA, P+A, P+R, R+A, S+B): ").strip().upper()
    try:
        market_line = float(input("Enter the market line (e.g., 25.5): ").strip())
    except ValueError:
        print("Invalid market line. Please enter a number.")
        exit()

    target_opp = input("Enter opponent abbreviation (e.g., CHI), or leave blank for all: ").strip().upper() or None
    home_filter = input("Only home games? (Y/N or blank for both): ").strip().lower()
    filter_home = True if home_filter == "y" else False if home_filter == "n" else None
    b2b_filter = input("Only back-to-back games? (Y/N or blank for both): ").strip().lower()
    filter_b2b = b2b_filter == "y"
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
                print(f"Scraped {len(season_games)} games for {player_id} in {season}")
                all_games.extend(season_games)
            except Exception as e:
                print(f"Failed to scrape {player_id} for {season}: {e}")
        
        print(f"\n--- Results for {player_id} ---")
        if all_games:
            avg_ev = calculate_ev(all_games, stat_key, market_line)
            print(f"Average EV for {stat_key} (Line {market_line}): {avg_ev:.2f}")
            for g in all_games:
                val = g[stat_key]
                if val is not None:
                    diff = val - market_line
                    p_win = 1 / (1 + math.exp(-0.5 * diff))
                    p_lose = 1 - p_win
                    ev = p_win - p_lose
                    print(f"Date: {g['date']} | Opp: {g['opponent']} | {stat_key}: {val} | MIN: {g['MIN']} | Result: {g['result']} | EV: {ev:.2f} | {'Home' if g['home'] else 'Away'} {'B2B' if g['b2b'] else ''}")
        else:
            print("No games found.")
