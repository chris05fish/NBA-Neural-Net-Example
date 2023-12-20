import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder

# Function to fetch NBA game data from the API
def fetch_nba_data(season):
    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = game_finder.get_data_frames()[0]
    # Extract relevant features and labels from the retrieved data
    features = games[['TEAM_ID', 'GAME_DATE', 'PTS', 'AST', 'REB']]  # Modify based on your needs
    labels = games['WL'].apply(lambda x: 1 if x == 'W' else 0)  # Convert 'W'/'L' to binary labels
    return features, labels

# Example: Fetch NBA data for the 2021-2022 season
season = '2021-22'
features, labels = fetch_nba_data(season)

# Create your custom dataset
class NBADataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        numeric_features = features[['PTS', 'AST', 'REB']]  # Adjust based on your selected features
        self.features = torch.tensor(numeric_features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

nba_dataset = NBADataset(features, labels)
nba_dataloader = DataLoader(nba_dataset, batch_size=10, shuffle=True)

class NBAWinPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NBAWinPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Assuming you have features with appropriate dimensions for your NBA dataset
input_size = len(features.columns) - 2
hidden_size = 64  # Adjust as needed
output_size = 2  # Binary classification: win or lose

nba_model = NBAWinPredictor(input_size, hidden_size, output_size)
optimizer = optim.Adam(nba_model.parameters(), lr=0.001)

# Training loop (modify as needed)
EPOCHS = 3
for epoch in range(EPOCHS):
    for data in nba_dataloader:
        features, labels = data
        optimizer.zero_grad()
        output = nba_model(features.float())  # Ensure features are converted to float
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    print(loss)