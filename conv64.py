import torch
from torch import nn


class Conv64F(nn.Module):
    def __init__(
        self,
        is_flatten=False,
        is_feature=False,
        leaky_relu=False,
        negative_slope=0.2,
        last_pool=True,
        maxpool_last2=True,
    ):
        super(Conv64F, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature
        self.last_pool = last_pool
        self.maxpool_last2 = maxpool_last2

        if leaky_relu:
            activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        else:
            activation = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )
        self.layer3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation,
        )
        self.layer4_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)

        if self.maxpool_last2:
            out3 = self.layer3_maxpool(out3)  # for some methods(relation net etc.)

        out4 = self.layer4(out3)
        if self.last_pool:
            out4 = self.layer4_pool(out4)

        if self.is_flatten:
            out4 = out4.view(out4.size(0), -1)

        if self.is_feature:
            return out1, out2, out3, out4

        return out4

class SingleLinear(nn.Module):
    def __init__(self):
        super(SingleLinear, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(64*5*5, 64),
        )
    
    def forward(self, x):
        output = self.stack(x)
        return output


class ReparaDecoder(nn.Module):
    def __init__(self):
        super(ReparaDecoder, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(64*5*5, 10),
            nn.LeakyReLU(inplace=True)
        )
        self.log_var = nn.Sequential(
            nn.Linear(64*5*5, 10),
            nn.LeakyReLU(inplace=True)
        )
        
        self.stack = nn.Sequential(
            nn.Linear(10, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 84*84*3),
            nn.Tanh()
        )
    
    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        # reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps*std + mu

        out = self.stack(z)
        return mu, log_var, out


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 84*84
            nn.Flatten(),
            nn.Linear(84**2, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            # output is (nc) x 256
        )

        self.decoder = nn.Sequential(
            # input is (nc) x 256
            nn.Linear(256, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 84**2),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = out.view(3, 84, 84)
        return out
