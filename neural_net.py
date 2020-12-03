# %% Torch Imports
import torch
import torch.nn as nn

# %% Neural Net building blocks
class Residual(nn.Module):
    def __init__(self) -> None:
        super(Residual, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.norm(out)
        out += x
        out = self.relu(out)
        return out

class ValueHead(nn.Module):
    def __init__(self) -> None:
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = out.view(-1, 64)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out

class PolicyHead(nn.Module):
    def __init__(self) -> None:
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(256, 100, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(100)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(100 * 64, 8*8*64)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = out.view(-1, 100 * 64)
        out = self.fc(out)
        out = self.softmax(out)
        return out


class AlphaZeroChess(nn.Module):
    def __init__(self) -> None:
        super(AlphaZeroChess, self).__init__()
        # self.conv = nn.Conv2d(13, 256, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(14, 256, kernel_size=3, stride=1, padding=1)
        self.res = Residual()
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()
    
    def forward(self, board):
        x = self.conv(board)
        # for _ in range(8):
        for _ in range(19):
            x = self.res(x)
        p = self.policy_head(x)
        p = p.view(-1, 64, 8, 8)
        v = self.value_head(x)
        return p, v


# %%
