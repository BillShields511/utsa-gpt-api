from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time


def scrape_current_page(driver, all_games, clean_games):
    days = driver.find_elements(By.CLASS_NAME, "Table__Title")
    tables = driver.find_elements(By.CLASS_NAME, "Table__TBODY")

    for i in range(len(days)):
        date = days[i].text
        rows = tables[i].find_elements(By.TAG_NAME, "tr")

        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 2:
                game_data = {
                    "Date": date,
                    "Away": cols[0].text,
                    "Home": cols[1].text
                }
                all_games.append(game_data)
    clean_data(all_games, clean_games)


def clean_data(all_games, clean_games):
    for game in all_games:
        g_day = game["Date"]
        day, date = g_day.split(", ", 1)
        date = date.replace(", 2025", "").strip()

        home_team = game["Home"].replace("@", "").replace("\n", "").strip()
        clean_games.append({
            "Day" : day.strip(),
            "Date": date.strip(),
            "Away": game["Away"].strip(),
            "Home": home_team
        })


if __name__ == "__main__":
    service = Service(executable_path="./chromedriver")
    driver = webdriver.Chrome(service=service)
    driver.get("https://www.espn.com/nfl/schedule/_/week/3/year/2025/seasontype/2")

    all_games = []
    clean_games = []
    weeks_to_scrape = 1

    for _ in range(weeks_to_scrape):
        time.sleep(2)
        scrape_current_page(driver, all_games, clean_games)

        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button.Arrow--right"))
            )
            next_button.click()
        except Exception:
            print("No more weeks.")
            break

    df = pd.DataFrame(clean_games)
    df.to_csv("nfl_schedule.csv", mode="a", header=False, index=False)
    driver.quit()
