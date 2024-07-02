import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



##########   Load and Preprocess the Data   ##########
# Load the data
data = pd.read_csv('MyMLProjects/MSFT.csv', parse_dates=['Date'])
data = data[(data['Date'] >= '2006-01-01') & (data['Date'] <= '2020-12-31')]
data = data[['Date', 'Close']]

# Normalize the data
data['Close'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()

# Convert to numpy array
prices = data['Close'].values



##########   Create a Dataset Class   ##########
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
        
seq_length = 10
train_indices = (data['Date'] < '2017-01-01').values
test_indices = (data['Date'] >= '2017-01-01').values

train_dataset = StockDataset(prices[train_indices], seq_length)
test_dataset = StockDataset(prices[test_indices], seq_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



##########   Build the LSTM Model   ##########
class StockPricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(StockPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockPricePredictor().to(device)



##########   Train the Model   ##########
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(-1)  # Add feature dimension
        outputs = model(inputs).squeeze(-1)  # Remove extra dimension
        
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')



##########   Make Predictions   ##########
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in DataLoader(test_dataset, batch_size=1, shuffle=False):
        inputs = inputs.to(device).unsqueeze(-1)
        prediction = model(inputs).squeeze(-1)
        predictions.append(prediction.item())
        actuals.append(targets.item())

predictions = np.array(predictions)
actuals = np.array(actuals)



##########   Plot the Results   ##########
plt.figure(figsize=(14,7))

# Plot the actual values
plt.plot(data['Date'][test_indices][seq_length:], actuals, label='Actual')

# Plot the predicted values
plt.plot(data['Date'][test_indices][seq_length:], predictions, label='Predicted')

plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.legend()
plt.show()

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
