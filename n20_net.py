import torch.nn as nn
import torch.nn.functional as F


class N20Net(nn.Module):
    def __init__(self, input_size):
        super(N20Net, self).__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 27),
            nn.LeakyReLU(0.2),
            nn.Linear(27,16),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)

        )

    def forward(self, x):
        x = self.net(x)
        return x