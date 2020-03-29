from torch import nn

class Net(nn.Module):
    def __init__(self,n_actions=144):
        super(Net, self).__init__()
        base_out_channels = 4
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(1, base_out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(base_out_channels)
        self.conv2 = nn.Conv2d(base_out_channels, base_out_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_out_channels*2)
        self.conv3 = nn.Conv2d(base_out_channels*2, base_out_channels*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_out_channels*4)
        self.conv4 = nn.Conv2d(base_out_channels*4, base_out_channels*8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_out_channels*8)
        self.conv5 = nn.Conv2d(base_out_channels*8, base_out_channels*4, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(base_out_channels*4)
        self.conv6 = nn.Conv2d(base_out_channels*4, base_out_channels*2, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(base_out_channels*2)
        self.conv7 = nn.Conv2d(base_out_channels*2, base_out_channels, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(base_out_channels)
        self.linear1 = nn.Linear(17*17*base_out_channels, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.steps_done = 0
        self.wins = 0

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))

        x = self.flatten(x)
        x = self.relu(self.bn8(self.linear1(x)))
        x = self.linear2(x)
        # X now has the q_values
        return x