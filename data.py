import pandas as pd
import numpy as np
import node
from sklearn.preprocessing import StandardScaler

team_locations = {
    "Arizona": 1,
    "Atlanta": 2,
    "Baltimore": 3,
    "Buffalo": 4,
    "Carolina": 5,
    "Chicago": 6,
    "Cincinnati": 7,
    "Cleveland": 8,
    "Dallas": 9,
    "Denver": 10,
    "Detroit": 11,
    "Green Bay": 12,
    "Houston": 13,
    "Indianapolis": 14,
    "Jacksonville": 15,
    "Kansas City": 16,
    "Las Vegas": 17,
    "LA Chargers": 18,
    "LA Rams": 19,
    "Miami": 20,
    "Minnesota": 21,
    "New England": 22,
    "New Orleans": 23,
    "NY Giants": 24,
    "NY Jets": 25,
    "Philadelphia": 26,
    "Pittsburgh": 27,
    "San Francisco": 28,
    "Seattle": 29,
    "Tampa Bay": 30,
    "Tennessee": 31,
    "Washington": 32
}

def get_matrix(g):
    arr = []
    for team in g.keys():
        for name, weight in g[team].connections:
            arr.append(weight)
    matrix = np.array(arr).reshape(32, 32)
    return(matrix)

def normalize_matrix(matrix):
    column_sums = matrix.sum(axis=0)  # Sum along each column
    normalized_matrix = matrix / column_sums

    return normalized_matrix

def massey_matrix(data):
    arr = []
    for (t1, t2, diff) in data:
        for i in range(1, 34):
            if i == t1:
                arr.append(1)
            elif i == t2:
                arr.append(-1)
            elif i == 33:
                arr.append(diff)
            else:
                arr.append(0)
    matrix = np.array(arr).reshape(272, 33)
    A = matrix[:, :-1]
    b = matrix[:, -1]
    return A, b

def massey_method(A, b):
    new_A = A.T @ A
    new_b = A.T @ b

    new_A[-1] = 1
    new_b[-1] = 0

    x = np.linalg.solve(new_A, new_b)
    return x


def main():

    # Load the CSV file
    df = pd.read_csv('nfl_scores.csv')
    df = df.drop_duplicates()

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

    team_losses_dict = total_losses.to_dict()

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

    df['Differential'] = -(abs(df['Score 1'] - df['Score 2']))

    teams = np.sort(df['Team 1'].unique())

    g: dict[str, node.Node] = {}

    for team in teams:
        g[team] = node.Node(team, total_losses[team], total_losses_diff[team])

    for team in teams:
        g[team].add_random(g.values())

    massey_data = []

    for index, row in df.iterrows():
        if row['Score 1'] > row['Score 2']:
            loser = g[row['Team 2']]
            winner = g[row['Team 1']]
            loser.add_connection(winner, row['Differential'])
            massey_data.append(((team_locations[row['Team 2']]), (team_locations[row['Team 1']]), (row['Differential'])))
        elif row['Score 1'] < row['Score 2']:
            loser = g[row['Team 1']]
            winner = g[row['Team 2']]
            loser.add_connection(winner, row['Differential'])
            massey_data.append(((team_locations[row['Team 1']]), (team_locations[row['Team 2']]), (row['Differential'])))

    matrix = get_matrix(g)
    A = normalize_matrix(matrix)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Find the index where the eigenvalue is 1
    index = np.isclose(eigenvalues, 1)

    # Extract the eigenvector corresponding to eigenvalue 1
    if np.any(index):
        eigenvector_for_1 = eigenvectors[:, index][..., 0]
        vec = eigenvector_for_1.real
    else:
        print("Eigenvalue 1 not found.")

        # Get the indices of the largest 5 values
    indices = np.argsort(vec)[-5:]  # Get indices of the largest 5 elements

    # Get the largest 5 values
    largest_values = vec[indices]

    print('PageRank rankings:')
    # Print the largest 5 values and their indices
    for idx, val in zip(indices, largest_values):
        print(f"Index: {idx}, Value: {val}")

    A, b = massey_matrix(massey_data)
    sol = massey_method(A, b)
    print(sol)

    indices = np.argsort(sol)[-5:]

    largest_values = sol[indices]

    print('Massey rankings:')


    # Print the largest 5 values and their indices
    for idx, val in zip(indices, largest_values):
        print(f"Index: {idx}, Value: {val}")

   # Step 1: Load the data
    teamstats_df = pd.read_csv('teamstats.csv')

    # Step 2: Remove the 'team' column
    teamstats_df = teamstats_df.drop(columns=['team'])

    # Convert percentage columns to numeric
    percent_cols = ['rzp', 'orzp', 'fgp']  # Specify the columns with percentages
    for col in percent_cols:
        teamstats_df[col] = teamstats_df[col].str.replace('%', '').astype(float) / 100

    # Convert 'atop' column to seconds
    teamstats_df['atop'] = pd.to_timedelta(teamstats_df['atop'])
    teamstats_df['atop'] = teamstats_df['atop'].dt.total_seconds()

    # Step 3: Scale stats
    scaler = StandardScaler()
    normalized_stats = scaler.fit_transform(teamstats_df)

    # Step 4: Append all remaining column entries to a list
    data_list = normalized_stats.tolist()

    # Step 5: Convert the list to a NumPy matrix
    stat_matrix = np.array(data_list).reshape(32, 8)  # Assuming 32 teams and 8 stats

    # Step 6: Solve the matrix equation (stat_matrix * weights = rankings)
    # Solve for the weights using numpy.linalg.solve
    pr_sol = np.linalg.lstsq(stat_matrix, vec)
    mm_sol = np.linalg.lstsq(stat_matrix, sol)

    print("Solutions for PageRank:", pr_sol[0] * 500)
    print("Solutions for Massey:", mm_sol[0])

if __name__=="__main__":
    main()