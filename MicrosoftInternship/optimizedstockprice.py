# 1) Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import optuna

# Check if CUDA is available
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Load and Preprocess the Data

# Load the data
data = pd.read_csv('MyMLProjectsMSFT.csv', parse_dates=['Date'])
data = data[(data['Date'] >= '1987-01-01') & (data['Date'] <= '2021-12-31')]
data = data[['Date', 'Close']]

# Normalize and convert to numpy array
prices = ((data['Close'] - data['Close'].mean()) / data['Close'].std()).values

# Define the training and testing periods
training_start = '1987-01-01'
training_end = '2009-12-31'
test_start = '2010-01-01'
test_end = '2021-12-31'

# Create train and test indices
train_indices = ((data['Date'] >= training_start) & (data['Date'] <= training_end)).values
test_indices = ((data['Date'] >= test_start) & (data['Date'] <= test_end)).values

# 3) Create a Dataset Class

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length]
        y = self.data[index+self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
# 4) Define the LSTM Model

class StockPricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden1_size=256, hidden2_size=256, dense1_size=128, dense2_size=64):
        super(StockPricePredictor, self).__init__()
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        
        self.lstm1 = nn.LSTM(input_size, hidden1_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1_size, hidden2_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden2_size, dense1_size)
        self.fc2 = nn.Linear(dense1_size, dense2_size)
        self.fc3 = nn.Linear(dense2_size, 1)
        
    def forward(self, x):
        h0_1 = torch.zeros(1, x.size(0), self.hidden1_size).to(x.device)
        c0_1 = torch.zeros(1, x.size(0), self.hidden1_size).to(x.device)
        
        out, _ = self.lstm1(x, (h0_1, c0_1))
        
        h0_2 = torch.zeros(1, out.size(0), self.hidden2_size).to(x.device)
        c0_2 = torch.zeros(1, out.size(0), self.hidden2_size).to(x.device)
        
        out, _ = self.lstm2(out, (h0_2, c0_2))
        
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
# 5) Define the Objective Function for Optuna

def objective(trial):
    # Suggest values for the hyperparameters
    seq_length = trial.suggest_int('seq_length', 10, 60)
    hidden1_size = trial.suggest_int('hidden1_size', 32, 256)
    hidden2_size = trial.suggest_int('hidden2_size', 32, 256)
    dense1_size = trial.suggest_int('dense1_size', 32, 128)
    dense2_size = trial.suggest_int('dense2_size', 16, 64)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(prices[train_indices], seq_length)
    test_dataset = StockDataset(prices[test_indices], seq_length)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize the model with suggested hyperparameters
    model = StockPricePredictor(1, hidden1_size, hidden2_size, dense1_size, dense2_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training the model
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(-1)
            outputs = model(inputs).squeeze(-1)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(-1)
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    return val_loss

# 6) Optimize Hyperparameters with Optuna

# Optimize hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best parameters
print('Best parameters: ', study.best_params)

# 7) Plot Hyperparameter Optimization History

# Plot optimization history
fig = plt.figure(figsize=(14, 7))
plt.plot(study.trials_dataframe().number, study.trials_dataframe().value)
plt.xlabel('Trial')
plt.ylabel('Validation Loss')
plt.title('Optimization History')
plt.show()

# 8) Train the Model with Optimized Parameters

# Extract optimized parameters
best_params = study.best_params
seq_length = best_params['seq_length']
hidden1_size = best_params['hidden1_size']
hidden2_size = best_params['hidden2_size']
dense1_size = best_params['dense1_size']
dense2_size = best_params['dense2_size']
num_epochs = best_params['num_epochs']
learning_rate = best_params['learning_rate']

# Create datasets and dataloaders
train_dataset = StockDataset(prices[train_indices], seq_length)
test_dataset = StockDataset(prices[test_indices], seq_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model with optimized parameters
model = StockPricePredictor(1, hidden1_size, hidden2_size, dense1_size, dense2_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(-1)
        outputs = model(inputs).squeeze(-1)
        
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(-1)
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')

# 9) Plot Training and Validation Loss

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 10) Make Predictions on Training and Test Data

# Make predictions on the training data
model.eval()
train_predictions = []
train_actuals = []

with torch.no_grad():
    for inputs, targets in DataLoader(train_dataset, batch_size=1, shuffle=False):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(-1)
        outputs = model(inputs).squeeze(-1)
        train_predictions.append(outputs.item())
        train_actuals.append(targets.item())

# Make predictions on the test data
test_predictions = []
test_actuals = []

with torch.no_grad():
    for inputs, targets in DataLoader(test_dataset, batch_size=1, shuffle=False):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(-1)
        outputs = model(inputs).squeeze(-1)
        test_predictions.append(outputs.item())
        test_actuals.append(targets.item())

# 11) Plot Actual vs Predicted Prices

# Convert normalized values back to original scale
train_actuals = np.array(train_actuals) * data['Close'].std() + data['Close'].mean()
train_predictions = np.array(train_predictions) * data['Close'].std() + data['Close'].mean()
test_actuals = np.array(test_actuals) * data['Close'].std() + data['Close'].mean()
test_predictions = np.array(test_predictions) * data['Close'].std() + data['Close'].mean()

# Create date ranges for plotting
train_dates = data['Date'][train_indices][seq_length:]
test_dates = data['Date'][test_indices][seq_length:]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(train_dates, train_actuals, label='Actual Training Prices')
plt.plot(train_dates, train_predictions, label='Predicted Training Prices')
plt.plot(test_dates, test_actuals, label='Actual Test Prices')
plt.plot(test_dates, test_predictions, label='Predicted Test Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()

