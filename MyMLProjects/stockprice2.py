import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Load the data
data = pd.read_csv('MyMLProjects/MSFT.csv')

# Display the first few rows of the data
print(data.head())

# Filter data between 2010 and 2020
data['Date'] = pd.to_datetime(data['Date'])
data = data[(data['Date'] >= '2010-01-01') & (data['Date'] <= '2020-12-31')]

# Select the 'Close' column and convert to numpy array
closing_prices = data['Close'].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices = closing_prices.reshape(-1, 1)
scaled_prices = scaler.fit_transform(closing_prices)

# Split the data into training and test sets
train_size = int(len(scaled_prices) * 0.8)
train_data, test_data = scaled_prices[:train_size], scaled_prices[train_size:]

# Create sequences of 60 days of data for training
def create_sequences(data, seq_length=60):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

train_sequences, train_labels = create_sequences(train_data)
test_sequences, test_labels = create_sequences(test_data)

# Convert sequences to PyTorch tensors
train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_sequences = torch.tensor(test_sequences, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Create DataLoader for batching
batch_size = 64
train_dataset = TensorDataset(train_sequences, train_labels)
test_dataset = TensorDataset(test_sequences, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.ReLU(50, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Initialize the model, define the loss function and the optimizer
model = StockPredictor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print the model architecture
print(model)

# Training the model
epochs = 50
for epoch in range(epochs):
    model.train()
    for seq, label in train_loader:
        seq, label = seq.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
test_predictions = []
with torch.no_grad():
    for seq, _ in test_loader:
        seq = seq.to(device)
        output = model(seq)
        test_predictions.extend(output.cpu().numpy())

# Inverse transform the predictions and the actual values
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
actual_values = scaler.inverse_transform(test_data[60:])

# Plot the predictions and the actual values
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
