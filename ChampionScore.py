# Import Dependencies
import pandas as pd
import numpy as np
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

def get_team_stats_data(year: int) -> pd.DataFrame:
    # Pause for 3.5 seconds to avoid getting blocked (20 requests/minute limit)
    time.sleep(3.5)

    # Set the URL for season standings page
    url = f'https://www.basketball-reference.com/leagues/NBA_{year}.html'

    # Get the HTML content of the page
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Get Columns to Use for each table
    columns_params = {
        'rate_stats': ['Team', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'],
        'advanced_stats': ['Team', 'Age', 'W', 'L', 'PW', 'PL', 'MOV', 'SOS', 'SRS', 'ORtg', 'DRtg', 'NRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'Team eFG%', 'Team TOV%', 'Team ORB%', 'Team FT/FGA', 'Opponent eFG%', 'Opponent TOV%', 'Opponent DRB%', 'Opponent FT/FGA', 'Arena', 'Attend.', 'Attend./G'],
        'team_shooting_stats': ['Team', 'G', 'MP', 'FG%', 'Dist.', '%FGA 2P', '%FGA 0-3', '%FGA 3-10', '%FGA 10-16', '%FGA 16-3P', '%FGA 3P', '2P%', '0-3%', '3-10%', '10-16%', '16-3P%', '3P%', '2P Assisted %', '3P Assisted %', '%FGA Dunks', 'MD% Dunks', '%FGA Layup', 'MD% Layups', '%3PA Corner', '3P% Corner', 'Heaves Att.', 'Heaves Made'],
        'opponent_shooting_stats': ['Team', 'G', 'MP', 'FG%', 'Dist.', '%FGA 2P', '%FGA 0-3', '%FGA 3-10', '%FGA 10-16', '%FGA 16-3P', '%FGA 3P', '2P%', '0-3%', '3-10%', '10-16%', '16-3P%', '3P%', '2P Assisted %', '3P Assisted %', '%FGA Dunks', 'MD% Dunks', '%FGA Layup', 'MD% Layups', '%3PA Corner', '3P% Corner'],
        'team_shooting_stats_pre_2002': ['Team', 'G', 'MP', 'FG%', 'Dist.', '%FGA 2P', '%FGA 0-3', '%FGA 3-10', '%FGA 10-16', '%FGA 16-3P', '%FGA 3P', '2P%', '0-3%', '3-10%', '10-16%', '16-3P%', '3P%', '2P Assisted %', '3P Assisted %', '%FGA Dunks', 'MD% Dunks', '%3PA Corner', '3P% Corner', 'Heaves Att.', 'Heaves Made'],
        'opponent_shooting_stats_pre_2002': ['Team', 'G', 'MP', 'FG%', 'Dist.', '%FGA 2P', '%FGA 0-3', '%FGA 3-10', '%FGA 10-16', '%FGA 16-3P', '%FGA 3P', '2P%', '0-3%', '3-10%', '10-16%', '16-3P%', '3P%', '2P Assisted %', '3P Assisted %', '%FGA Dunks', 'MD% Dunks', '%3PA Corner', '3P% Corner']
    }

    # Get Standings Data & Combine dataframes & Clean the Team name column
    eastern_conference_standings_df = get_standings_data(soup, year, 'Eastern')
    western_conference_standings_df = get_standings_data(soup, year, 'Western')
    league_standings_df = pd.concat([eastern_conference_standings_df, western_conference_standings_df], ignore_index=True)
    league_standings_df = league_standings_df.sort_values(by='W/L%', ascending=False).reset_index(drop=True)
    
    # Get Team & Opponent Stats Data (Rate, Shooting, and Advanced Stats)
    team_rate_stats_df = get_team_stats(soup, year, 'team', 'per_poss-team', 'rate', columns_params['rate_stats'])
    team_advanced_stats_df = get_team_stats(soup, year, 'team', 'advanced-team', 'advanced', columns_params['advanced_stats'])
    opponent_rate_stats_df = get_team_stats(soup, year, 'opponent', 'per_poss-opponent', 'rate', columns_params['rate_stats'])
    if year >= 1997:
        if year < 2002:
            team_shooting_stats_df = get_team_stats(soup, year, 'team', 'shooting-team', 'shooting', columns_params['team_shooting_stats_pre_2002'])
            opponent_shooting_stats_df = get_team_stats(soup, year, 'opponent', 'shooting-opponent', 'shooting', columns_params['opponent_shooting_stats_pre_2002'])
        else:
            team_shooting_stats_df = get_team_stats(soup, year, 'team', 'shooting-team', 'shooting', columns_params['team_shooting_stats'])
            opponent_shooting_stats_df = get_team_stats(soup, year, 'opponent', 'shooting-opponent', 'shooting', columns_params['opponent_shooting_stats'])

    # Drop unnecessary columns from the dataframes
    team_advanced_stats_df = team_advanced_stats_df.drop(columns=['W', 'L', 'SRS'])
    opponent_rate_stats_df = opponent_rate_stats_df.drop(columns=['G', 'MP', 'Team/Opponent'])
    if year >= 1997:
        team_shooting_stats_df = team_shooting_stats_df.drop(columns=['G', 'MP', 'FG%', 'Team/Opponent'])
        opponent_shooting_stats_df = opponent_shooting_stats_df.drop(columns=['G', 'MP', 'FG%', 'Team/Opponent'])

    # Clean the Team name column for each dataframe
    if year >= 1997:
        league_standings_df, team_rate_stats_df, team_shooting_stats_df, team_advanced_stats_df, opponent_rate_stats_df, opponent_shooting_stats_df = clean_team_name_column([league_standings_df, team_rate_stats_df, team_shooting_stats_df, team_advanced_stats_df, opponent_rate_stats_df, opponent_shooting_stats_df])
    else:
        league_standings_df, team_rate_stats_df, team_advanced_stats_df, opponent_rate_stats_df = clean_team_name_column([league_standings_df, team_rate_stats_df, team_advanced_stats_df, opponent_rate_stats_df])
    
    # Add custom suffixes to all column names that would otherwise be duplicated
    if year >= 1997:
        team_rate_stats_df, team_shooting_stats_df, team_advanced_stats_df, opponent_rate_stats_df, opponent_shooting_stats_df = add_suffix([team_rate_stats_df, team_shooting_stats_df, team_advanced_stats_df, opponent_rate_stats_df, opponent_shooting_stats_df], ['_Per100_Team', '_Shooting', '_Advanced', '_Per100_Opponent', '_Shooting_Opponent'])
    else:
        team_rate_stats_df, team_advanced_stats_df, opponent_rate_stats_df = add_suffix([team_rate_stats_df, team_advanced_stats_df, opponent_rate_stats_df], ['_Per100_Team', '_Advanced', '_Per100_Opponent'])

    # Merge the dataframes
    df = pd.merge(league_standings_df, team_rate_stats_df, on=['Team', 'Year'])
    df = pd.merge(df, team_advanced_stats_df, on=['Team', 'Year'])
    df = pd.merge(df, opponent_rate_stats_df, on=['Team', 'Year'])
    if year >= 1997:
        df = pd.merge(df, team_shooting_stats_df, on=['Team', 'Year'])
        df = pd.merge(df, opponent_shooting_stats_df, on=['Team', 'Year'])
    
    # Get the playoff odds data
    odds_df = get_odds_data(year - 1, year)

    # Merge the odds data with the main dataframe
    df = pd.merge(df, odds_df, on='Team', how='left')

    return df

def get_game_logs_data(year: int, team_abbreviations: dict) -> pd.DataFrame:
    inverted_team_abbreviations = {v: k for k, v in team_abbreviations.items()}
    df_combined_regular_season = pd.DataFrame()
    df_combined_playoffs = pd.DataFrame()

    for abbreviation in team_abbreviations.values():
        # Pause for 3.5 seconds to avoid getting blocked (20 requests/minute limit)
        time.sleep(3.5)

        print(f'Getting game logs for {abbreviation} in {year}')

        # Set the URL for the season game logs page
        url = f'https://www.basketball-reference.com/teams/{abbreviation}/{year}_games.html'

        # Get the HTML content of the page
        response = requests.get(url)
        if response.status_code == 404:
            print(f'Page not found for {abbreviation} in {year}. Skipping...')
            continue
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table with the game logs data
        table = soup.find('table', {'id': 'games'})
        rows = table.find_all('tr')
        data = []

        # Extract the data from the table
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append(cols)
        df_regular_season = pd.DataFrame(data, columns=['G', 'Date', 'Start (ET)', 'Box Score Link', 'Home/Road', 'Opponent', 'Result', 'OT', 'Tm', 'Opp', 'W', 'L', 'Streak', 'Notes'])
        df_regular_season = df_regular_season.dropna(subset=['G'])

        # Add the Team/Year to the dataframe
        df_regular_season['Team'] = inverted_team_abbreviations[abbreviation]
        df_regular_season['Year'] = year

        # Combine the dataframes for each team
        df_combined_regular_season = pd.concat([df_combined_regular_season, df_regular_season], ignore_index=True)

        # Find the table with the playoff game logs data (if it exists)
        table = soup.find('table', {'id': 'games_playoffs'})
        if table is not None:
            rows = table.find_all('tr')
            data = []

            # Extract the data from the table
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append(cols)
            df_playoffs = pd.DataFrame(data, columns=['G', 'Date', 'Start (ET)', 'Box Score Link', 'Home/Road', 'Opponent', 'Result', 'OT', 'Tm', 'Opp', 'W', 'L', 'Streak', 'Notes'])
            df_playoffs = df_playoffs.dropna(subset=['G'])

            # Add the Team/Year to the dataframe
            df_playoffs['Team'] = inverted_team_abbreviations[abbreviation]
            df_playoffs['Year'] = year

            # Combine the dataframes for each team
            df_combined_playoffs = pd.concat([df_combined_playoffs, df_playoffs], ignore_index=True)

    return df_combined_regular_season, df_combined_playoffs

def get_standings_data(soup: BeautifulSoup, year: int, conference: str) -> pd.DataFrame:
    # Find the standings table for the appropriate conference
    table_id = 'divs_standings_E' if conference == 'Eastern' else 'divs_standings_W'
    table = soup.find('table', {'id': table_id})
    rows = table.find_all('tr')
    data = []

    # Extract the data from the table
    for row in rows:
        team = row.find('th').text.strip()
        if 'Division' not in team and 'Conference' not in team:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append([team] + cols)
    df = pd.DataFrame(data, columns=['Team', 'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS'])
    df['Year'] = year
    df['Conference'] = 'Eastern' if conference == 'Eastern' else 'Western'
    df['Made Playoffs'] = df['Team'].apply(lambda x: 1 if '*' in x else 0)
    
    return df

def get_team_stats(soup: BeautifulSoup, year: int, team_or_opponent: str, table_id: str, table_type: str, columns: list) -> pd.DataFrame:
    # Find the appropriate table
    table = soup.find('table', {'id': table_id})
    rows = table.find_all('tr')
    data = []

    # Extract the data from the table
    for row in rows:
        rank = row.find('th').text.strip()
        if rank == 'Rk' or rank == '':
            continue
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols if ele.text.strip() != '']
        data.append(cols)
    df = pd.DataFrame(data, columns=columns)
    df['Year'] = year
    if table_type != 'advanced':
        df['Team/Opponent'] = 'Team' if team_or_opponent == 'team' else 'Opponent'
    
    return df

def get_odds_data_html(url, retries=3, backoff_factor=0.5):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX or 5XX
            return response
        except ConnectionError as e:
            print(f"Connection failed: {e}. Retrying...")
            time.sleep(backoff_factor * (2 ** attempt))
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            break
    return None

def get_odds_data(start_year: int, end_year: int) -> pd.DataFrame:
    url = f'https://www.sportsoddshistory.com/nba-main/?y={start_year}-{end_year}&sa=nba&a=finals&o=r1'

    # Get the HTML content of the page
    response = get_odds_data_html(url)
    if response:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table with the odds data
        table = soup.find('table', {'class': 'soh1'})

        # Extract the data from the table
        data = []

        # Find the index of the "Round 1" column
        headers = table.find_all('th')
        round_1_index = None
        counter = 0
        for i, header in enumerate(headers):
            counter += 1
            if '...' in header.text:
                counter -= 1
            if 'Round 1' in header.text:
                round_1_index = counter - len([th for th in headers[:i] if th.has_attr('rowspan')])  # Adjust for rowspan
                break

        # Add the data to the list if the column was found
        if round_1_index is not None:
            for row in table.find_all('tbody')[0].find_all('tr'):
                cols = row.find_all('td')
                team_name = cols[0].text.strip()
                round_1_data = cols[round_1_index].text.strip()
                data.append((team_name, round_1_data))
        else:
            print("Pre-Playoff odds not found.")

        # Create a dataframe from the data
        df = pd.DataFrame(data, columns=['Team', 'Odds'])

        return df

def add_suffix(dfs: list[pd.DataFrame], suffix: list[str]) -> list[pd.DataFrame]:
    current_index = 0

    for df in dfs:
        for column in df.columns:
            if column not in ['Team', 'Year', 'G', 'MP', 'Team/Opponent']:
                dfs[current_index] = dfs[current_index].rename(columns={column: column + suffix[current_index]})
        current_index += 1

    return dfs

def clean_team_name_column(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    for df in dfs:
        df['Team'] = (df['Team'].str.replace(r'\(\d+\)', '', regex=True)
                                .str.replace('(', '')
                                .str.replace(')', '')
                                .str.replace('*', '')
                                .str.rstrip())
    return dfs

def fill_in_custom_feature_columns(df: pd.DataFrame, df_game_log_regular_season: pd.DataFrame, df_game_log_playoffs: pd.DataFrame, year: int) -> pd.DataFrame:
    # Fill in the following custom feature columns:
    # - Record vs .600+ teams
    # - Record in clutch games (games decided by 5 points or less)
    # - Number of playoff wins in the previous year (converted to championship share [i.e. 16 wins = 1.0])
    df['Record vs .600+ Teams'] = 0.0
    df['Number of Playoff Wins'] = 0
    df['Champion Share'] = 0.0

    # Ensure important columns is numeric
    df['W/L%'] = pd.to_numeric(df['W/L%'], errors='coerce')
    df_game_log_regular_season['Tm'] = pd.to_numeric(df_game_log_regular_season['Tm'], errors='coerce')
    df_game_log_regular_season['Opp'] = pd.to_numeric(df_game_log_regular_season['Opp'], errors='coerce')

    # Calculate the record vs .600+ teams (in the regular season)
    teams_600_plus = df[(df['W/L%'] >= 0.600) & (df['Year'] == year)]['Team'].values    
    for team in df['Team'].unique():
        # Filter game logs for current team & year against .600+ teams
        team_games_vs_600_plus = df_game_log_regular_season[(df_game_log_regular_season['Team'] == team) & 
                                                            (df_game_log_regular_season['Opponent'].isin(teams_600_plus)) & 
                                                            (df_game_log_regular_season['Year'] == year)]
        
        # Calculate win rate
        if not team_games_vs_600_plus.empty:
            wins = team_games_vs_600_plus['Result'].value_counts().get('W', 0)
            total_games = len(team_games_vs_600_plus)
            win_rate = wins / total_games if total_games > 0 else 0.0
        else:
            win_rate = 0.0
        
        # Assign win rate to the team in df
        df.loc[(df['Team'] == team) & (df['Year'] == year), 'Record vs .600+ Teams'] = win_rate
    
    # Calculate Record in Clutch Games (in the regular season)
    df_game_log_regular_season['Is Clutch'] = abs(df_game_log_regular_season['Tm'] - df_game_log_regular_season['Opp']) <= 5
    df_game_log_regular_season['Clutch Win'] = (df_game_log_regular_season['Is Clutch']) & (df_game_log_regular_season['Tm'] > df_game_log_regular_season['Opp'])
    clutch_win_rates = df_game_log_regular_season.groupby(['Team', 'Year']).apply(lambda x: x['Clutch Win'].sum() / x['Is Clutch'].sum() if x['Is Clutch'].sum() > 0 else 0.0).reset_index(name='Record in Clutch Games')
    df = df.merge(clutch_win_rates, on=['Team', 'Year'], how='left')
    df_game_log_regular_season.drop(columns=['Is Clutch', 'Clutch Win'], inplace=True)
    
    # Calculate the number of playoff wins in the previous year
    if year > 2002:
        previous_year_df = df_game_log_playoffs[df_game_log_playoffs['Year'] == year - 1]
        if not previous_year_df.empty:
            playoff_wins = int(previous_year_df.iloc[-1]['W']) # This line in isolation gets the playoff wins for the current year (when previous_year_df is replaced with df_game_log_playoffs. However, current overall code doesn't return correct value.)
        else:
            playoff_wins = 0
        
        print(playoff_wins) # See right-hanging comment above. Also keep an eye on the Record in Clutch Games column. It appears to be appending another version of itself (i.e. _x _y versions of the columns) each year.

        #playoff_wins = df_game_log_playoffs[(df_game_log_playoffs['Year'] == year - 1) & (df_game_log_playoffs['Result'] == 'W')]['Number of Playoff Wins'].sum()
        champion_share = playoff_wins / 16 if year - 1 > 2002 else playoff_wins / 15
        df.loc[df['Year'] == year, 'Number of Playoff Wins'] = playoff_wins
        df.loc[df['Year'] == year, 'Champion Share'] = champion_share

    return df

def scrape_and_save_data(df: pd.DataFrame) -> pd.DataFrame:
    team_abbreviations = {'Sacromento Kings': 'SAC'}

    # team_abbreviations = {
    #     'Atlanta Hawks': 'ATL',
    #     'Boston Celtics': 'BOS',
    #     'Brooklyn Nets': 'BRK',
    #     'Charlotte Bobcats': 'CHA',
    #     'Charlotte Hornets': 'CHO',
    #     'Chicago Bulls': 'CHI',
    #     'Cleveland Cavaliers': 'CLE',
    #     'Dallas Mavericks': 'DAL',
    #     'Denver Nuggets': 'DEN',
    #     'Detroit Pistons': 'DET',
    #     'Golden State Warriors': 'GSW',
    #     'Houston Rockets': 'HOU',
    #     'Indiana Pacers': 'IND',
    #     'Los Angeles Clippers': 'LAC',
    #     'Los Angeles Lakers': 'LAL',
    #     'Memphis Grizzlies': 'MEM',
    #     'Miami Heat': 'MIA',
    #     'Milwaukee Bucks': 'MIL',
    #     'Minnesota Timberwolves': 'MIN',
    #     'New Jersey Nets': 'NJN',
    #     'New Orleans Hornets': 'NOH',
    #     'New Orleans Pelicans': 'NOP',
    #     'New York Knicks': 'NYK',
    #     'Oklahoma City Thunder': 'OKC',
    #     'Orlando Magic': 'ORL',
    #     'Philadelphia 76ers': 'PHI',
    #     'Phoenix Suns': 'PHO',
    #     'Portland Trail Blazers': 'POR',
    #     'Sacramento Kings': 'SAC',
    #     'San Antonio Spurs': 'SAS',
    #     'Seattle SuperSonics': 'SEA',
    #     'Toronto Raptors': 'TOR',
    #     'Utah Jazz': 'UTA',
    #     'Washington Wizards': 'WAS'
    # }

    # Scrape data for each year (if no Excel file exists in folder, pausing to avoid getting blocked [20 requests/minute limit])
    for year in range(2002, 2004): # 2025
        print(f'Getting data for {year}')
        
        # Get Team Data
        df = pd.concat([df, get_team_stats_data(year)])

        # Get Game Logs Data
        df_game_logs_regular_season, df_game_logs_playoffs = get_game_logs_data(year, team_abbreviations)

        # Filter Regular Season Game Logs for calculated features
        df = fill_in_custom_feature_columns(df, df_game_logs_regular_season, df_game_logs_playoffs, year)

        print(df)


        # Get Player Stats Data (League Leaders, Award Winners, etc.)

    # Save to Excel
    df.to_excel('NBA_Team_Stats.xlsx', index=False)

    return df

def prepare_data_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary rows/columns
    df = df.drop(columns=['Conference', 'G', 'MP', 'W', 'L', 'GB', 'PS/G', 'PA/G', 'Team/Opponent', 'Heaves Att._Shooting', 'Heaves Made_Shooting', 'Arena_Advanced', 'Attend._Advanced', 'Attend./G_Advanced', 'FG%_Per100_Team', 'FG%_Per100_Opponent', 'PW_Advanced', 'PL_Advanced', 'MOV_Advanced', 'SOS_Advanced', 'Age_Advanced', 'Pace_Advanced'])
    df = df[df.columns.drop(list(df.filter(regex='_Shooting|3P%|2P%|FT%|_Per100')))]

    # Scale the data
    df['Year Copy'] = df['Year'] # Create a copy of the Year column to group by to keep in line with pandas new groupby behavior (avoids deprecation warning)
    df = df.groupby('Year Copy').apply(scale_group, include_groups=False)

    # Eliminate non-playoff teams
    df = df[df['Made Playoffs'] == 1]
    df = df.drop(columns=['Made Playoffs'])

    # Find Implied Probabilities from Odds
    df['Implied Odds'] = df['Odds'].apply(odds_to_probability)
    df = df.drop(columns=['Odds'])

    # Add a column for the target variable (Won Championship)
    # Won Championship = 1 if team won the championship that year, 0 otherwise
    df['Won Championship'] = 0
    championship_teams = {
        1980: 'Los Angeles Lakers',
        1981: 'Boston Celtics',
        1982: 'Los Angeles Lakers',
        1983: 'Philadelphia 76ers',
        1984: 'Boston Celtics',
        1985: 'Los Angeles Lakers',
        1986: 'Boston Celtics',
        1987: 'Los Angeles Lakers',
        1988: 'Los Angeles Lakers',
        1989: 'Detroit Pistons',
        1990: 'Detroit Pistons',
        1991: 'Chicago Bulls',
        1992: 'Chicago Bulls',
        1993: 'Chicago Bulls',
        1994: 'Houston Rockets',
        1995: 'Houston Rockets',
        1996: 'Chicago Bulls',
        1997: 'Chicago Bulls',
        1998: 'Chicago Bulls',
        1999: 'San Antonio Spurs',
        2000: 'Los Angeles Lakers',
        2001: 'Los Angeles Lakers',
        2002: 'Los Angeles Lakers',
        2003: 'San Antonio Spurs',
        2004: 'Detroit Pistons',
        2005: 'San Antonio Spurs',
        2006: 'Miami Heat',
        2007: 'San Antonio Spurs',
        2008: 'Boston Celtics',
        2009: 'Los Angeles Lakers',
        2010: 'Los Angeles Lakers',
        2011: 'Dallas Mavericks',
        2012: 'Miami Heat',
        2013: 'Miami Heat',
        2014: 'San Antonio Spurs',
        2015: 'Golden State Warriors',
        2016: 'Cleveland Cavaliers',
        2017: 'Golden State Warriors',
        2018: 'Golden State Warriors',
        2019: 'Toronto Raptors',
        2020: 'Los Angeles Lakers',
        2021: 'Milwaukee Bucks',
        2022: 'Golden State Warriors',
        2023: 'Denver Nuggets'
    }
    for year, team in championship_teams.items():
        df.loc[(df['Year'] == year) & (df['Team'] == team), 'Won Championship'] = 1

    return df

# Function to apply scaling within each group
def scale_group(group):
    scaler = MinMaxScaler(feature_range=(0, 1))
    columns_to_scale = ['ORtg_Advanced', 'DRtg_Advanced', 'NRtg_Advanced', 'FTr_Advanced', '3PAr_Advanced', 'TS%_Advanced', 'Team eFG%_Advanced', 'Team TOV%_Advanced', 'Team ORB%_Advanced', 'Team FT/FGA_Advanced', 'Opponent eFG%_Advanced', 'Opponent TOV%_Advanced', 'Opponent DRB%_Advanced', 'Opponent FT/FGA_Advanced']
    columns_to_invert = ['DRtg_Advanced', 'Team TOV%_Advanced', 'Opponent eFG%_Advanced', 'Opponent DRB%_Advanced', 'Opponent FT/FGA_Advanced']

    # Scale columns
    group[columns_to_scale] = scaler.fit_transform(group[columns_to_scale])
    
    # Invert scaling for specified columns where lower values are better
    for column in columns_to_invert:
        group[column] = 1 - group[column]
    
    return group

def odds_to_probability(odds: str) -> float:
    if odds > 0:
        probability = 100 / (odds + 100)
    else:
        probability = (-odds) / ((-odds) + 100)
    
    return probability

def get_train_test_data(df: pd.DataFrame, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split the data into training and testing sets
    train_df = df[df['Year'] < test_year]
    test_df = df[df['Year'] == test_year]

    X_train = train_df.drop(columns=['Team', 'Year', 'Won Championship'])
    y_train = train_df['Won Championship']
    X_test = test_df.drop(columns=['Team', 'Year', 'Won Championship'])
    y_test = test_df['Won Championship']

    return train_df, test_df, X_train, y_train, X_test, y_test

def train_model(df: pd.DataFrame, test_year: int, model_type: str) -> tuple[str, bool]:
    train_df, test_df, X_train, y_train, X_test, y_test = get_train_test_data(df, test_year)

    # Select and create the model based on model_type
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'svm':
        model = SVC(probability=True)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier()
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_type == 'neural_network':
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    elif model_type == 'naive_bayes':
        model = GaussianNB()
    elif model_type == 'ada_boost':
        model = AdaBoostClassifier()
    else:
        raise ValueError("Invalid model type. Choose a different model.")

    # Train the model
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Get the team with the highest probability
    max_proba_index = np.argmax(y_pred_proba)
    predicted_champion = test_df.iloc[max_proba_index]['Team']

    # Check if the prediction was correct
    if test_year < 2024:
        actual_champion = test_df.loc[y_test == 1, 'Team'].values[0]
        prediction_correct = actual_champion == predicted_champion
    else:
        actual_champion = 'N/A'
        prediction_correct = False

    return predicted_champion, prediction_correct

def print_results(results):
    for year, result in results.items():
        print(f"Year: {year}, Predicted Champion: {result['Predicted Champion']}, Prediction Correct: {result['Prediction Correct']}")

    correct_predictions = sum([result['Prediction Correct'] for result in results.values()])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions
    print(f"Model Accuracy: {accuracy} ({correct_predictions}/{total_predictions}) [{results[2023]['Name']}]")
    print("\n")

def main():
    # Create a dataframe
    df = pd.DataFrame()

    # Load data from file if it exists, else scrape & save data from basketball-reference then load it
    file_exists = True if os.path.isfile('NBA_Team_Stats.xlsx') else False
    if file_exists == False:
        df = scrape_and_save_data(df)
    df = pd.read_excel('NBA_Team_Stats.xlsx')

    # Prep df for machine learning
    df = prepare_data_for_ml(df)

    # Save the prepped data to a new Excel file (if needed)
    prepped_file_exists = True if os.path.isfile('NBA_Team_Stats_Prepped.xlsx') else False
    if prepped_file_exists == False:
        df.to_excel('NBA_Team_Stats_Prepped.xlsx', index=False)

    # Train Models
    LR_results = {}
    RF_results = {}
    SVM_results = {}
    GB_results = {}
    KNN_results = {}
    DT_results = {}
    NN_results = {}
    NB_results = {}
    ADA_results = {}
    for year in range(2003, 2024):
        LR_champion, LR_prediction_correct = train_model(df, year, 'logistic_regression')
        RF_champion, RF_prediction_correct = train_model(df, year, 'random_forest')
        SVM_champion, SVM_prediction_correct = train_model(df, year, 'svm')
        GB_champion, GB_prediction_correct = train_model(df, year, 'gradient_boosting')
        KNN_champion, KNN_prediction_correct = train_model(df, year, 'knn')
        DT_champion, DT_prediction_correct = train_model(df, year, 'decision_tree')
        NN_champion, NN_prediction_correct = train_model(df, year, 'neural_network')
        NB_champion, NB_prediction_correct = train_model(df, year, 'naive_bayes')
        ADA_champion, ADA_prediction_correct = train_model(df, year, 'ada_boost')
        
        LR_results[year] = {'Predicted Champion': LR_champion, 'Prediction Correct': LR_prediction_correct, 'Name': 'Logistic Regression'}
        RF_results[year] = {'Predicted Champion': RF_champion, 'Prediction Correct': RF_prediction_correct, 'Name': 'Random Forest'}
        SVM_results[year] = {'Predicted Champion': SVM_champion, 'Prediction Correct': SVM_prediction_correct, 'Name': 'Support Vector Machine'}
        GB_results[year] = {'Predicted Champion': GB_champion, 'Prediction Correct': GB_prediction_correct, 'Name': 'Gradient Boosting'}
        KNN_results[year] = {'Predicted Champion': KNN_champion, 'Prediction Correct': KNN_prediction_correct, 'Name': 'K-Nearest Neighbors'}
        DT_results[year] = {'Predicted Champion': DT_champion, 'Prediction Correct': DT_prediction_correct, 'Name': 'Decision Tree'}
        NN_results[year] = {'Predicted Champion': NN_champion, 'Prediction Correct': NN_prediction_correct, 'Name': 'Neural Network'}
        NB_results[year] = {'Predicted Champion': NB_champion, 'Prediction Correct': NB_prediction_correct, 'Name': 'Naive Bayes'}
        ADA_results[year] = {'Predicted Champion': ADA_champion, 'Prediction Correct': ADA_prediction_correct, 'Name': 'AdaBoost'}
    
    # Print the results & accuracy of the models
    print_results(LR_results)
    print_results(RF_results)
    print_results(SVM_results)
    print_results(GB_results)
    print_results(KNN_results)
    print_results(DT_results)
    print_results(NN_results)
    print_results(NB_results)
    print_results(ADA_results)

if __name__ == '__main__':
    main()

# To-Do:
# - Modify the code so that you can track the accuracy of the model more nuacedly (e.g. points for guessing runner-up, etc.)
# - Add more features to the model (i.e. player award shares, top 3 in conference [bool, calculated feature], record vs .600+ teams, previous year playoff wins, etc.)