import torch.nn as nn
import torch.nn.functional as F


class CKDNet(nn.Module):
    def __init__(self, input_size):
        super(CKDNet, self).__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32,16),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 2)

        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)