import torch
import torch.nn as nn
import torch.nn.functional as F


class DimensionalityReducer(nn.Module):
    def __init__(self, in_channels=256, out_channels=4):
        super().__init__()
        # 使用 Conv2d 进行通道的降维
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接应用 Conv2d 处理四维输入 [batch_size, channels, height, width]
        if x.dim() < 4:
            x = x.unsqueeze(1)  # 在第二个维度插入
        x = self.conv2d(x)
        x = F.interpolate(x, size=(1024, 768), mode='bilinear', align_corners=False)
        return x
