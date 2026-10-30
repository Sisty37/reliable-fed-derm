"""EfficientNet-B0 baseline and weighted focal loss, as in FLamby."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# FLamby class weights (flamby/datasets/fed_isic2019/loss.py)
ALPHA = torch.tensor([5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443])


class Baseline(nn.Module):
    def __init__(self, pretrained=True, num_classes=8):
        super().__init__()
        self.base_model = (EfficientNet.from_pretrained("efficientnet-b0") if pretrained
                           else EfficientNet.from_name("efficientnet-b0"))
        self.base_model._fc = nn.Linear(self.base_model._fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=ALPHA, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma

    def forward(self, logits, targets):
        logpt = F.log_softmax(logits.float(), dim=1).gather(1, targets.view(-1, 1)).view(-1)
        pt = logpt.exp()
        at = self.alpha.to(logits.device).gather(0, targets.view(-1))
        return (-(1 - pt) ** self.gamma * logpt * at).mean()
