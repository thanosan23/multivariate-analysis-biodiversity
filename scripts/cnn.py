import torch
from torch import nn
import torch.nn.functional as F
from utils.read_dataset import read_dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

df = read_dataset(data={
    'biodiversity': ['Year', 'number of species']
})

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_length = 5
X, y = create_sequences(df['number of species'].values, seq_length)

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1,1))

X = torch.tensor(X).float()
y = torch.tensor(y).float()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 2)
        self.fc1 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = ConvNet()
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X.unsqueeze(1))
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

def predict(model, sequence):
    sequence = np.array(sequence)
    sequence = scaler.transform(sequence.reshape(-1, 1))
    sequence = torch.tensor(sequence).float()
    model.eval()
    sequence = sequence.reshape(-1, 1, 5)
    prediction = model(sequence)
    prediction = scaler.inverse_transform(prediction.detach().numpy())
    return prediction

print(predict(model, [631, 652, 634, 602, 600]))
