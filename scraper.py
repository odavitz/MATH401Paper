import requests
from bs4 import BeautifulSoup
import csv
import re

# Step 1: Send a GET request to the target URL
url = "https://www.footballdb.com/scores/index.html?lg=NFL&yr=2023&type=reg&wk=18"
# Mimic a browser by adding a User-Agent header
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Send the request with headers
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Step 2: Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    game_tables = soup.find_all('table', class_='fb_component_tbl darkhdr')

    # Step 3: Open a CSV file for writing
    with open('nfl_scores.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(['Team 1', 'Score 1', 'Team 2', 'Score 2'])  # Write the header row
        writer.writerow('\n')

        # Loop through each table (game)
        for table in game_tables:
            # Find all rows in the current table
            rows = table.find('tbody').find_all('tr')
            
            # Ensure there are exactly two rows (one for each team)
            if len(rows) == 2:
                # Extract details for the first team
                team1_cell = rows[0].find('td', class_='left')
                team1 = team1_cell.get_text(strip=True) if team1_cell else "Unknown"
                team1 = re.sub(r'\s*\(.*?\)', '', team1)  # Remove content in parentheses

                score1_cell = rows[0].find_all('td')[-1]
                score1 = score1_cell.get_text(strip=True) if score1_cell else "N/A"

                # Extract details for the second team
                team2_cell = rows[1].find('td', class_='left')
                team2 = team2_cell.get_text(strip=True) if team2_cell else "Unknown"
                team2 = re.sub(r'\s*\(.*?\)', '', team2)  # Remove content in parentheses

                score2_cell = rows[1].find_all('td')[-1]
                score2 = score2_cell.get_text(strip=True) if score2_cell else "N/A"

                # Write the game data to the CSV file
                writer.writerow([team1, score1, team2, score2])

    print("Data has been written to nfl_scores.csv")

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
