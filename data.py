import pandas as pd

# Load the CSV file
df = pd.read_csv('nfl_scores.csv')
df = df.drop_duplicates()

totalDiff = {
    "Altanta": -9
}

# Convert scores to numeric (in case there are any non-numeric characters)
df['Score 1'] = pd.to_numeric(df['Score 1'], errors='coerce')
df['Score 2'] = pd.to_numeric(df['Score 2'], errors='coerce')

# Create a column for losses for Team1 (where Team1's score is less than Team2's score)
df['Team1_Losses'] = df['Score 1'] < df['Score 2']

# Create a column for losses for Team2 (where Team2's score is less than Team1's score)
df['Team2_Losses'] = df['Score 2'] < df['Score 1']

# Group by Team1 and count losses
team1_losses = df.groupby('Team 1')['Team1_Losses'].sum()

# Group by Team2 and count losses
team2_losses = df.groupby('Team 2')['Team2_Losses'].sum()

# Combine the losses from Team1 and Team2
total_losses = team1_losses.add(team2_losses, fill_value=0)

team_losses = total_losses.to_dict()

df['Point_Differential1'] = df['Score 1'] - df['Score 2']
df['Point_Differential2'] = df['Score 2'] - df['Score 1']

# Create a column for losses for Team1 (where Team1's score is less than Team2's score)
df['Team1_Losses'] = df['Score 1'] < df['Score 2']

# Create a column for losses for Team2 (where Team2's score is less than Team1's score)
df['Team2_Losses'] = df['Score 2'] < df['Score 1']

# Group by Team1 and sum the point differentials for losses
team1_losses_diff = df[df['Team1_Losses']].groupby('Team 1')['Point_Differential1'].sum()

# Group by Team2 and sum the point differentials for losses
team2_losses_diff = df[df['Team2_Losses']].groupby('Team 2')['Point_Differential2'].sum()

# Combine the point differentials from Team1 and Team2
total_losses_diff = team1_losses_diff.add(team2_losses_diff, fill_value=0)

# Convert to a dictionary
losses_diff_dict = total_losses_diff.to_dict()
