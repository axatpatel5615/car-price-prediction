import pandas as pd
import numpy as np

# 1. Load the single master file from your dataset
df = pd.read_csv('IPL.csv')

# 2. Standardize team name changes/rebrands across history
team_mappings = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
}
df['batting_team'] = df['batting_team'].replace(team_mappings)
df['bowling_team'] = df['bowling_team'].replace(team_mappings)
df['match_won_by'] = df['match_won_by'].replace(team_mappings)

# Keep only active current teams to keep your UI dropdowns looking clean
active_teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bengaluru',
    'Kolkata Knight Riders', 'Rajasthan Royals', 'Sunrisers Hyderabad',
    'Delhi Capitals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
]
df = df[df['batting_team'].isin(active_teams) & df['bowling_team'].isin(active_teams)]

# 3. Filter down exclusively to the Second Innings (The Run Chase)
chase_df = df[df['innings'] == 2].copy()

# 4. Feature Engineering using pre-calculated columns from your dataset
# Convert total target and current state progress metrics
chase_df['runs_left'] = chase_df['runs_target'] - chase_df['team_runs']
chase_df['balls_left'] = 120 - chase_df['team_balls']
chase_df['wickets_left'] = 10 - chase_df['team_wicket']

# Calculate rolling mathematical rates
# (Using np.where to safely avoid zero-division errors on the first or final ball)
chase_df['crr'] = np.where(
    chase_df['team_balls'] > 0, 
    (chase_df['team_runs'] * 6) / chase_df['team_balls'], 
    0
)
chase_df['rrr'] = np.where(
    chase_df['balls_left'] > 0, 
    (chase_df['runs_left'] * 6) / chase_df['balls_left'], 
    0
)

# 5. Define target classification labels (1 if chasing team won, 0 if they lost)
chase_df['result'] = np.where(chase_df['batting_team'] == chase_df['match_won_by'], 1, 0)

# 6. Extract only the columns needed for training our ML Model
final_columns = [
    'batting_team', 'bowling_team', 'venue', 'runs_left', 
    'balls_left', 'wickets_left', 'runs_target', 'crr', 'rrr', 'result'
]
processed_data = chase_df[final_columns].dropna()

# Drop rows where balls_left calculation becomes zero or negative to keep statistical rates accurate
processed_data = processed_data[processed_data['balls_left'] > 0]

# Save the polished output file
processed_data.to_csv('ipl_processed_data.csv', index=False)
print(f"Dataset successfully prepared! Total training rows extracted: {processed_data.shape[0]}")