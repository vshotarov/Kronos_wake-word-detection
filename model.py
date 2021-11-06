import torch
from torch import nn
from torch.nn import functional as F


class WWDModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(WWDModel, self).__init__()

        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
                nn.Conv1d(81,81,10,2,padding=10//2),
                nn.MaxPool1d(10,2),
                nn.GELU(),
                nn.Dropout(.1),
                nn.BatchNorm1d(81),
                nn.Conv1d(81,64,4,2,padding=2),
                nn.MaxPool1d(4,2),
                nn.GELU(),
                nn.Dropout(.1),
                nn.BatchNorm1d(64),
                )

        self.dense = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1088, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Sigmoid(),
                #nn.Dropout(.1),
                )

        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.conv(x)
        x = x.transpose(1,2)
        x = self.dense(x)
        x = self.fc(x)
        return x

    def classify(self, x):
        outputs = self(x)
        probabilities = F.softmax(outputs, dim=1)[0]
        return torch.argmax(probabilities, dim=0)

