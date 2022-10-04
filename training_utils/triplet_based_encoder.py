from torch import nn, squeeze


class Big_Triplet_Encoder(nn.Module) :
    def __init__(self, latent_space_size=32, input_channels=5) :
        super(Big_Triplet_Encoder, self).__init__()

        self.input_channels = input_channels

        self.latent_space_size = latent_space_size

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, latent_space_size, 1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x) :
      x = self.encoder(x)
      x = squeeze(x)
      return x