import torch
import torch.nn as nn


class DualInputFusionBlock(nn.Module):
    def __init__(self, channels=64):
        super(DualInputFusionBlock, self).__init__()
        # Fusion mechanism for combining two input features
        self.fusion_conv = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, feat1, feat2):
        # Concatenate features from both inputs
        combined = torch.cat([feat1, feat2], dim=1)

        # Generate attention weights
        attention_weights = self.attention(combined)

        # Apply fusion
        fused = self.lrelu(self.fusion_conv(combined))

        # Apply attention-based weighting
        output = fused * attention_weights + feat1 * (1 - attention_weights)

        return output

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_rate, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_rate)
        self.rdb2 = ResidualDenseBlock(channels, growth_rate)
        self.rdb3 = ResidualDenseBlock(channels, growth_rate)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class DualInputESRGANGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=64, num_blocks=16, scale_factor=4):
        super(DualInputESRGANGenerator, self).__init__()
        self.scale_factor = scale_factor

        # Initial convolution for both inputs
        self.conv_first1 = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.conv_first2 = nn.Conv2d(in_channels, channels, 3, 1, 1)

        # Fusion block to combine initial features
        self.fusion_block = DualInputFusionBlock(channels)

        # RRDB blocks
        self.rrdb_blocks = nn.ModuleList([RRDB(channels) for _ in range(num_blocks)])

        # Final convolution for feature extraction
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)

        # Upsampling
        self.upconv1 = nn.Conv2d(channels, channels * 4, 3, 1, 1)  # 4x channels for pixel shuffle
        self.upconv2 = nn.Conv2d(channels, channels * 4, 3, 1, 1)  # 4x channels for pixel shuffle
        self.pixel_shuffle = nn.PixelShuffle(2)

        # Final output
        self.conv_last = nn.Conv2d(channels, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x1, x2):
        # Initial feature extraction for both inputs
        feat1 = self.conv_first1(x1)
        feat2 = self.conv_first2(x2)

        # Fuse features from both inputs
        fused_feat = self.fusion_block(feat1, feat2)
        body_feat = fused_feat

        # RRDB blocks
        for block in self.rrdb_blocks:
            body_feat = block(body_feat)

        # Residual connection
        body_feat = self.conv_body(body_feat)
        feat = fused_feat + body_feat

        # Upsampling (4x = 2x + 2x)
        feat = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))  # Now channels/4
        feat = self.lrelu(self.pixel_shuffle(self.upconv2(feat)))  # Now channels/4

        # Final output
        out = self.conv_last(feat)
        return out

class DualInputDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(DualInputDiscriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, stride=1, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(in_channels, base_channels, normalize=False),
            discriminator_block(base_channels, base_channels, stride=2),
            discriminator_block(base_channels, base_channels * 2),
            discriminator_block(base_channels * 2, base_channels * 2, stride=2),
            discriminator_block(base_channels * 2, base_channels * 4),
            discriminator_block(base_channels * 4, base_channels * 4, stride=2),
            discriminator_block(base_channels * 4, base_channels * 8),
            discriminator_block(base_channels * 8, base_channels * 8, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
