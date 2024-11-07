import torch
import torch.nn as nn
import numpy as np

from .utils import MSEDist

class ResidualBlock(nn.Module):
    def __init__(self, channels, act='silu', norm='layer'):
        super(ResidualBlock, self).__init__()
        self.act = nn.SiLU()
        if norm == 'layer':
            self.norm1 = nn.LayerNorm([channels, 1, 1])
            self.norm2 = nn.LayerNorm([channels, 1, 1])
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        out += residual
        return out

class ImageDecoderResnet(nn.Module):
    def __init__(self, shape, input_dim, 
                 depth=96, blocks=0, resize='stride', minres=4, sigmoid=False, 
                 act='silu', norm='layer'):
        super(ImageDecoderResnet, self).__init__()
        self.shape = shape  # 输出图像的形状，例如(64, 64, 3)
        self.input_dim = input_dim  # 输入特征的维度
        self.depth = depth  # 初始通道数，默认96
        self.blocks = blocks  # 每个阶段的残差块数量，默认0
        self.resize = resize  # 上采样方式，默认'stride'
        self.minres = minres  # 最小分辨率，默认4
        self.sigmoid = sigmoid  # 是否使用Sigmoid激活函数，默认False
        self.act = act  # 激活函数，默认'silu'
        self.norm = norm  # 归一化方式，默认'layer'

        # 计算需要的阶段数量
        self.stages = int(np.log2(self.shape[0]) - np.log2(self.minres))
        depth = self.depth * (2 ** (self.stages - 1))

        # 输入线性层，将输入特征映射到初始[minres, minres, depth]特征图上
        self.linear = nn.Linear(self.input_dim, self.minres * self.minres * depth)

        # 构建解码器的模块列表
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.stages):
            stage_modules = nn.ModuleList()
            # 添加残差块
            for j in range(self.blocks):
                stage_modules.append(ResidualBlock(depth, act=self.act, norm=self.norm))
            # 添加上采样层
            if i == self.stages - 1:
                # 最后一层，输出通道数为目标图像的通道数
                out_channels = self.shape[2] * 2
            else:
                out_channels = depth // 2
            if self.resize == 'stride':
                stage_modules.append(
                    nn.ConvTranspose2d(
                        in_channels=depth,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1
                    )
                )
            else:
                raise NotImplementedError(f"未实现的上采样方式：{self.resize}")
            self.decoder_blocks.append(stage_modules)
            depth = out_channels

    def forward(self, x):
        x = self.linear(x)
        depth = x.size(1) // (self.minres * self.minres)
        x = x.view(-1, depth, self.minres, self.minres)

        for stage_modules in self.decoder_blocks:
            for module in stage_modules:
                x = module(x)

        if x.size()[2:] != self.shape[:2]:
            x = nn.functional.interpolate(x, size=self.shape[:2], mode='bilinear', align_corners=False)
        x = x + 0.5
        mean, _ = torch.chunk(x, 2, dim=1)
        dist = MSEDist(mean, 3, "sum")
        return dist


if __name__ == "__main__":
    input_dim = 2048
    output_shape = (224, 224, 3)
    decoder = ImageDecoderResnet(
        shape=output_shape,
        input_dim=input_dim,
        depth=96,
        blocks=0,
        resize='stride',
        minres=4,
        sigmoid=False,
        act='silu',
        norm='layer'
    ).cuda()
    batch_size = 16
    input_tensor = torch.randn(batch_size, input_dim).cuda()

    output = decoder(input_tensor)
    print(output.shape)
