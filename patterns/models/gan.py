"""
Generative Adversarial Network
Implemented for Generative Replay POC.

This is a DCGan.
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    """ GAN for Generative Replay """

    def __init__(self, input_size, feature_size=32, channels=1):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_size,
                out_channels=feature_size * 4,
                kernel_size=4, 
                bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=feature_size,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.Tanh(),
        )     

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, features=32, channels=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features) x 16 x 16
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*2) x 8 x 8
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features*4) x 4 x 4
            nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":
    # Some simple tests for sanity check
    generator = Generator(100).apply(weights_init)
    z = torch.randn(1, 100, 1, 1)
    output = generator(z)
    print(output.size())

    discriminator = Discriminator().apply(weights_init)
    x = torch.randn(1, 1, 32, 32)
    output = discriminator(x)
    print(output.size())

