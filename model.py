import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels): #channels is 256 because we should be feeding the 256 x 8 x 8 into the resBlocks
        super().__init__()
        # We use padding=1 to keep the 8x8 spatial dimension the same
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # The Skip Connection logic
        return F.relu(out)
    

class Representation(nn.Module):
    def __init__(self, obsChannels, convChannels): #obsChannels = 119, convChannels = 256
        super().__init__()

        self.conv1 = nn.Conv2d(obsChannels, convChannels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(convChannels)
        self.relu = nn.ReLU()
        self.resblocks = nn.Sequential(*[ResBlock(convChannels) for _ in range(10)])

    def forward(self, x): #x is the initial observation block
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.resblocks(x)
        return x
    
class Dynamics(nn.Module):
    def __init__(self, stateChannels):
        super().__init__()


    def forward(self, actionChannels, stateChannels):        
        torch.cat((actionChannels, stateChannels), dim = 1)
