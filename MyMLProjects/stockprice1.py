# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('MyMLProjects/MSFT.csv')

# Display the first few rows of the data
print(data.head())

# Filter data between 2006 and 2017
data['Date'] = pd.to_datetime(data['Date'])
data = data[(data['Date'] >= '2006-01-01') & (data['Date'] <= '2017-12-31')]

# Select the 'Close' column and convert to numpy array
closing_prices = data['Close'].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices = closing_prices.reshape(-1, 1)
scaled_prices = scaler.fit_transform(closing_prices)

# Split the data into training and test sets
train_size = int(len(scaled_prices) * 0.8)
train_data, test_data = scaled_prices[:train_size], scaled_prices[train_size:]

# Create sequences of 20 days of data for training
def create_sequences(data, seq_length=20):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences

train_sequences = create_sequences(train_data)
test_sequences = create_sequences(test_data)

# Convert sequences to PyTorch tensors
train_sequences = [(torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)) for seq, label in train_sequences]
test_sequences = [(torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)) for seq, label in test_sequences]

class StockPredictor(nn.Module):
    def __init__(self):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Initialize the model, define the loss function and the optimizer
model = StockPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print the model architecture
print(model)

# Training the model
epochs = 100
for epoch in range(epochs):
    for seq, label in train_sequences:
        optimizer.zero_grad()
        output = model(seq.unsqueeze(0))
        loss = criterion(output, label.unsqueeze(0))
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
test_predictions = []
with torch.no_grad():
    for seq, _ in test_sequences:
        output = model(seq.unsqueeze(0))
        test_predictions.append(output.item())

# Inverse transform the predictions and the actual values
test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
actual_values = scaler.inverse_transform(test_data[20:])

# Plot the predictions and the actual values
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
