import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class SegmentationNN(pl.LightningModule):
    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))
        for param in self.model.parameters():
            param.requires_grad = False
        self.added = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Upsample((100,100)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Upsample((240, 240)),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.model(x)
        x = self.added(x)

        return x
