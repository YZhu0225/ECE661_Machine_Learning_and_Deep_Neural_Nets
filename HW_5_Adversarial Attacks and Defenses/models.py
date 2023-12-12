import torch
import torch.nn as nn

class NetA(nn.Module):
    def __init__(self,num_classes=10):
        super(NetA, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), # 28 x 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1), # 14 x 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), # 7 x 7
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*128, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out

class NetB(nn.Module):
    def __init__(self,num_classes=10):
        super(NetB, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), # 28 x 28
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,1,1), # 28 x 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1), # 14 x 14
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1), # 14 x 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2), # 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*64, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out
