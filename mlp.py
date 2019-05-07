import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size1, hidden_size2, out_features):
        super(MLP, self).__init__()
        self.hid1 = nn.Linear(in_features, hidden_size1)
        self.hid2 = nn.Linear(hidden_size1, hidden_size2)
        self.out = nn.Linear(hidden_size2, out_features)

    def forward(self, x):
        x = F.relu(self.hid1(x))
        x = F.relu(self.hid2(x))
        x = self.out(x)
        return x


