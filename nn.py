import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nba_api.stats.endpoints import leaguegamefinder

# Function to fetch NBA game data from the API
def fetch_nba_data(season):
    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = game_finder.get_data_frames()[0]
    # Extract relevant features and labels from the retrieved data
    features = games[['TEAM_ID', 'GAME_DATE', 'PTS', 'AST', 'REB']]  
    labels = games['WL'].apply(lambda x: 1 if x == 'W' else 0)  # Convert 'W'/'L' to binary labels
    return features, labels

# Example: Fetch NBA data for the 2021-2022 season
season = '2021-22'
features, labels = fetch_nba_data(season)

# Create custom dataset
class NBADataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        numeric_features = features[['PTS', 'AST', 'REB']]  # Adjust based on your selected features
        self.features = torch.tensor(numeric_features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

# Split the dataset into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Create DataLoader for the training set
train_dataset = NBADataset(features_train, labels_train)
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Create DataLoader for the testing set
test_dataset = NBADataset(features_test, labels_test)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Define your model, optimizer, and loss function
input_size = len(features.columns) - 2
hidden_size = 128  
output_size = 2  # Binary classification: win or lose

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

nba_model = NBAWinPredictor(input_size, hidden_size, output_size)
optimizer = optim.Adam(nba_model.parameters(), lr=0.001)

# Training loop 
EPOCHS = 10
for epoch in range(EPOCHS):
    nba_model.train()  # Set the model to training mode
    for data in train_dataloader:
        features, labels = data
        optimizer.zero_grad()
        output = nba_model(features.float())
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the testing set
    nba_model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in test_dataloader:
            features, labels = data
            output = nba_model(features.float())
            _, predictions = torch.max(output, 1)
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    # Calculate accuracy on the testing set
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}, Accuracy: {accuracy}")
