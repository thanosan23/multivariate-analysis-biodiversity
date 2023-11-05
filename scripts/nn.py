from pandas.io.formats.style import plt
from utils.read_dataset import read_dataset

from sklearn.preprocessing import  StandardScaler

import torch
import torch.nn as nn
import numpy as np

import shap
# from sklearn.linear_model import LinearRegression

df = read_dataset(data={
        'biodiversity': ['Year', 'number of species'],
        'carbon_dioxide': ['Year', 'Annual'],
        'pollution': ['Year', 'Volatile Organic Compounds'],
}, aggregate_col='Year')

scalers = {}
for col in ['number of species', 'Annual', 'Volatile organic compounds']:
    std = StandardScaler()
    df[col] = std.fit_transform(df[col].values.reshape(-1, 1))
    scalers[col] = std

training_data = df[:int(len(df)*0.8)]
testing_data = df[int(len(df)*0.8):]

X = df[['Year', 'Annual', 'Volatile organic compounds']]
y = df['number of species']

X_train = training_data[['Year', 'Annual', 'Volatile organic compounds']]
y_train = training_data['number of species']

X_test = testing_data[['Year', 'Annual', 'Volatile organic compounds']]
y_test = testing_data['number of species']

X, y = np.array(X), np.array(y)
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = LSTMModel(3, 32, 2)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1000)

for epoch in range(500):
    inputs = torch.from_numpy(X.astype(np.float32))
    labels = torch.from_numpy(y.astype(np.float32))

    outputs = model(inputs)
    optimizer.zero_grad()
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f'epoch {epoch+1}, loss {loss.item():.4f}')

def predict(model, x, y, z):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.array([[x, y, z]]).astype(np.float32))
        output = model(x)
        output = output.item()
        output = scalers['number of species'].inverse_transform([[output]])
        return output[0]

print(predict(model, 2050, 412, 0.5))
