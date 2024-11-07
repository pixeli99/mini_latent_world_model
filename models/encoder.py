import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=nn.SiLU(), norm='layer'):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels) if norm == 'layer' else nn.Identity()
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ImageEncoderResnet(nn.Module):
    def __init__(self, in_channels, depth=96, blocks=0, resize='stride', minres=4, act=nn.SiLU(), norm='layer'):
        super(ImageEncoderResnet, self).__init__()
        self.blocks = blocks
        self.depth = depth
        self.resize = resize
        self.minres = minres
        self.act = act

        self.res_blocks = nn.ModuleList()
        current_depth = in_channels
        
        # Calculate the number of stages based on input size and minres
        self.stages = int(torch.log2(torch.tensor(128)) - torch.log2(torch.tensor(minres)))  # Assuming input size is 128x128

        for i in range(self.stages):
            # Downsampling block with kernel size and stride as in JAX
            if resize == 'stride':
                self.res_blocks.append(Conv2dBlock(current_depth, depth, kernel_size=4, stride=2, padding=1, act=act, norm=norm))
            elif resize == 'stride3':
                stride = 3 if i == 0 else 2
                kernel_size = 5 if i == 0 else 4
                self.res_blocks.append(Conv2dBlock(current_depth, depth, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, act=act, norm=norm))
            elif resize == 'mean':
                self.res_blocks.append(Conv2dBlock(current_depth, depth, kernel_size=3, stride=1, padding=1, act=act, norm=norm))
            elif resize == 'max':
                self.res_blocks.append(Conv2dBlock(current_depth, depth, kernel_size=3, stride=1, padding=1, act=act, norm=norm))
            else:
                raise NotImplementedError(f"Resize method {resize} is not implemented.")
            
            # Add residual blocks within each stage
            for j in range(blocks):
                self.res_blocks.append(self._make_residual_block(depth, act, norm))
            
            current_depth = depth
            depth *= 2  # Double depth for next stage
        
        self.fc1 = nn.Linear(8192, 4096)

    def _make_residual_block(self, channels, act, norm):
        return nn.Sequential(
            Conv2dBlock(channels, channels, kernel_size=3, stride=1, padding=1, act=act, norm=norm),
            Conv2dBlock(channels, channels, kernel_size=3, stride=1, padding=1, act=act, norm=norm)
        )

    def forward(self, x):
        x = x - 0.5  # Normalizing the input as in JAX
        for i, block in enumerate(self.res_blocks):
            if isinstance(block, Conv2dBlock):
                x = block(x)
            else:
                residual = x
                x = block(x)
                x += residual  # Adding residual connection
                x = self.act(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    import torch
    batch_size = 4
    in_channels = 3
    height, width = 128, 128

    input_data = torch.randn(batch_size, in_channels, height, width)

    encoder = ImageEncoderResnet(in_channels=in_channels, depth=32, blocks=0, resize='stride', minres=4)
    encoded_output = encoder(input_data)

    print("Encoded Output Shape:", encoded_output.shape)
