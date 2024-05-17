# coding:utf-8
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from mamba_ssm import Mamba

'''
1. conda create -n mamba python=3.10

2. conda activate mamba

3. conda install cudatoolkit==11.7 -c nvidia

4. pip install torch==1.13 torchvision==0.14.0 torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu117

5. conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc

6. pip install packaging

7. pip install causal-conv1d

8. pip install mamba-ssm

9. pip install timm
'''

class Light_VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 16,
        drop_path: float = 0,
        d_state: int = 16,
        d_conv: int = 4, 
        expand: int = 2,
        norm_layer = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ln_1 = norm_layer(hidden_dim)

        if norm_layer is not None:
            self.ln_1 = norm_layer(hidden_dim)

        self.mamba = Mamba(
            d_model=hidden_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        if input.dtype == torch.float16:
            input = input.type(torch.float32)
        B, C = input.shape[:2]
        assert C == self.hidden_dim
        n_tokens = input.shape[2:].numel()
        img_dims = input.shape[2:]
        input_flat = input.reshape(B, C, n_tokens).transpose(-1, -2)
        input_norm = self.ln_1(input_flat)

        x1, x2, x3, x4 = torch.chunk(input_norm, 4, dim=2)
        x_mamba1 = x1 + self.drop_path(self.mamba(x1))
        x_mamba2 = x2 + self.drop_path(self.mamba(x2))
        x_mamba3= x3 + self.drop_path(self.mamba(x3))
        x_mamba4 = x4 + self.drop_path(self.mamba(x4))
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)
        out = x_mamba.transpose(-1, -2).reshape(B, self.hidden_dim, *img_dims)
        return out

if __name__ == "__main__":
    test = Light_VSSBlock(hidden_dim = 16,
        drop_path = 0,
        d_state = 16,
        d_conv = 4, 
        expand = 2).to("cuda")
    inputs = torch.rand(1,16,768).to("cuda")
    ans = test(inputs)
    print(ans)