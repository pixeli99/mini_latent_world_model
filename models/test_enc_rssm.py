import torch
import torch.nn as nn

from encoder import ImageEncoderResnet
from rssm import RSSM
# 设置模型参数
encoder = ImageEncoderResnet(in_channels=3, depth=32, blocks=0, resize='stride', minres=4, act=nn.SiLU(), norm='layer').cuda()
rssm = RSSM(deter=512, stoch=256, classes=None, initial="learned", unimix=0.01, action_clip=1.0, winit='normal', fan='avg', units=512).cuda()

# 设置测试输入
batch_size = 16
seq_len = 64
input_image = torch.randn(batch_size, 3, 128, 128).cuda()  # 模拟输入图像 (batch_size, channels, height, width)
action = torch.randn(batch_size, seq_len, 2).cuda()  # 模拟动作 (batch_size, seq_len, action_dim)
embed = encoder(input_image)  # 编码图像以获取嵌入向量

initial_state = rssm.initial_state(batch_size)

# 测试 RSSM 的 observe 方法
is_first = torch.zeros(batch_size, seq_len).bool()  # 模拟 is_first 标志
post, prior = rssm.observe(embed.unsqueeze(1).repeat(1, seq_len, 1), action, is_first, initial_state)

# 检查输出维度和内容
print("Post state shapes:")
for k, v in post.items():
    print(f"{k}: {v.shape}")

print("\nPrior state shapes:")
for k, v in prior.items():
    print(f"{k}: {v.shape}")

# 测试 imagine 方法
imagine_state = rssm.imagine(action, initial_state)
print("\nImagine state shapes:")
for k, v in imagine_state.items():
    print(f"{k}: {v.shape}")