import torchvision
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*list(vgg.children())[:35]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.features(input)
        target_features = self.features(target)
        return F.mse_loss(input_features, target_features)

class DualInputConsistencyLoss(nn.Module):
    def __init__(self):
        super(DualInputConsistencyLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, input1, input2):
        # Upscale both inputs to match output resolution
        upscaled_input1 = F.interpolate(input1, scale_factor=4, mode='bicubic', align_corners=False)
        upscaled_input2 = F.interpolate(input2, scale_factor=4, mode='bicubic', align_corners=False)

        # Ensure output preserves information from both inputs
        loss1 = self.mse_loss(output, upscaled_input1)
        loss2 = self.mse_loss(output, upscaled_input2)

        return (loss1 + loss2) / 2
