import torch 
from torch import nn 
from einops import rearrange 
from layers.spectral_convs import SpectralConv_2D_diag 
import numpy as np
import torch.nn.functional as F


def _pad_2_3_rule(x):
    orig_h, orig_w = x.shape[1], x.shape[2]
    pad_h = int((2 * orig_h) / 3)
    pad_w = int((2 * orig_w) / 3)
    if pad_h == 0 and pad_w == 0:
        return x, orig_h, orig_w
    x = rearrange(x, 'B H W C -> B C H W')
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    x = rearrange(x, 'B C H W -> B H W C')
    return x, orig_h, orig_w


def _unpad_2_3_rule(x, orig_h, orig_w):
    return x[:, :orig_h, :orig_w, :]

class HO_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, order=2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.order = order
        self.proj = nn.Conv2d(in_channels, out_channels * order, kernel_size=1, bias=True)
        
        self.spec_convs = SpectralConv_2D_diag(in_channels, out_channels, modes1, modes2)
        

    def forward(self, x):
        # x: [B, H, W, C]
        x = rearrange(x, 'B H W C -> B C H W')

        x = self.proj(x)

        x_list = torch.chunk(x, self.order, dim=1)
        z = x_list[0]
        for i in range(1, self.order):
            x_2 = x_list[i]
            z = z * x_2
        
        z = self.spec_convs(z)
        
        out = rearrange(z, 'B C H W -> B H W C')
        return out
    
class RMSNorm2D(nn.Module):
    def __init__(self, channels, name=None, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.name = name
        self.eps = eps
        self.log_rms = True

    def forward(self, x):
        # x: [B, H, W, C]
        rms = torch.mean(x*x, dim=-1, keepdim=True) 
            
        x = x / torch.sqrt(rms + self.eps)
        return x * self.weight
    
class DropPath(nn.Module):
    def __init__(self, p=0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x * mask / keep_prob
    
class HO_block(nn.Module):
    def __init__(self, width, modes1, modes2, order, name1, name2, p_drop_MLP, drop_path, expansion_MLP, two_thirds_zero_padding=False):
        super().__init__()
        self.HO_conv = HO_Conv(width, width, modes1, modes2, order)
        self.norm1 = RMSNorm2D(width, name1) 
        self.norm2 = RMSNorm2D(width, name2)  
        self.MLP = nn.Sequential(nn.Linear(width, width * expansion_MLP),
                                 nn.GELU(),
                                 nn.Dropout(p_drop_MLP),
                                 nn.Linear(width * expansion_MLP, width),
                                 nn.Dropout(p_drop_MLP))
        self.drop_path = DropPath(drop_path)
        self.two_thirds_zero_padding = two_thirds_zero_padding

    def forward(self, x):
        x_conv = self.norm1(x)
        if self.two_thirds_zero_padding:
            x_conv, orig_h, orig_w = _pad_2_3_rule(x_conv)
            x_conv = self.HO_conv(x_conv)
            x_conv = _unpad_2_3_rule(x_conv, orig_h, orig_w)
        else:
            x_conv = self.HO_conv(x_conv)

        x = x + self.drop_path(x_conv)
        x = x + self.drop_path(self.MLP(self.norm2(x)))
        return x
    
class HO_FNO(nn.Module): 
    def __init__(self, width, depth, in_channels, out_channels, modes1, modes2, order, p_drop_MLP, drop_path_rate, expansion_MLP, two_thirds_zero_padding=False):
        super().__init__()
        self.D = 2
        self.depth = depth

        self.linear_p = nn.Linear(in_channels, width)
        drop_rates = torch.linspace(0, drop_path_rate, depth)

        self.HO_blocks = nn.ModuleList([HO_block(width, modes1, modes2, order, name1=f"norm pre conv {l}", name2=f"norm pre mlp {l}", p_drop_MLP=p_drop_MLP, drop_path=drop_rates[l], expansion_MLP=expansion_MLP, two_thirds_zero_padding=two_thirds_zero_padding)
                                   for l in range(depth)])


        self.linear_q = nn.Linear(width, out_channels)
        
    def forward(self, x, fx=None):
        # x: [B, H, W, C] (Channel last)
        if fx is None:
            fx = self.get_grid(x.shape, x.device)
        x = torch.cat([x, fx], dim=-1)

        x = self.linear_p(x) # lifting

        # Apply the Fourier layers sequentially
        for block in self.HO_blocks:
            x = block(x)

        x = self.linear_q(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
        
