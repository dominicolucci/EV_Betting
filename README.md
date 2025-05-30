# EV_Betting: NBA Player Prop Betting Analysis

This project evaluates expected value (EV) for NBA player prop bets using historical game logs and machine‑learning models (Logistic Regression and Random Forest). It includes data scraping, preprocessing, modeling, and EV calculation across multiple validation strategies.



---

## Project Structure
---

## Key Features

- **Logistic Regression Modeling** for prop‑line EV evaluation  
- **Rolling, Date‑based, and Season‑based Splits** using `model_util.py`  
- **Custom Data Scraping** from Basketball Reference via `nba_api_scraper.py`  
- **Player Mapping Dictionary** to standardize inputs  
- **Back‑to‑Back Game Evaluator** and team‑stat tracking  
- **Automated Logging** of predictions and results to CSV  
- **Modular Design** for extensibility across models or sports  

---

## Example Usage

Run a model and calculate EV:

~~~bash
python logistic_regression_v2.py
# or
python main.py
~~~

Log EV and predictions:

~~~bash
python log_model_props.py
~~~

Scrape player data:

~~~bash
python nba_api_scraper.py
~~~

---

## Output Files

| File            | Purpose                                                                  |
|-----------------|--------------------------------------------------------------------------|
| `bets_log.csv`  | Logs: date, player, stat, market line, predicted probability, EV, result |
| `stats_log.csv` | Game‑by‑game stats with context flags (home/away, B2B, etc.)             |

---

## Dependencies

- `scikit‑learn`  
- `numpy`  
- `pandas`  
- `requests` / `beautifulsoup4`  
- Standard library (`datetime`, `csv`, `os`, etc.)

Install everything:

~~~bash
pip install -r requirements.txt
~~~

Generate a fresh `requirements.txt`:

~~~bash
pip freeze > requirements.txt
~~~

---

## TODO

- Integrate GUI or web dashboard  
- Add confidence intervals to model predictions  
- Expand to additional stat types (rebounds, assists, combos)  
- Improve depth‑chart / injury integration  
- Import sportsbook odds for real‑time value comparison

---

## **NOTE**

This project is still under active development. Features and functionality may change as it evolves.

