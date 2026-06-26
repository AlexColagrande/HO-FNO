"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities import *
from torch.optim import AdamW

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True

torch.cuda.set_device(1)

import torch 
from torch import nn 
from einops import rearrange 
from layers.spectral_convs import SpectralConv_3D_diag

class HO_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, order=2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.order = order

        #self.proj = nn.Conv3d(in_channels, out_channels * order, kernel_size=1, bias=True)
        self.spec_convs = SpectralConv_3D_diag(in_channels, out_channels, modes1, modes2, modes3)
        

    def forward(self, x):
        # x: [B, H, W, C]
        x = rearrange(x, 'B H W D C -> B C H W D')

        #x = self.proj(x)

        # x_list = torch.chunk(x, self.order, dim=1)
        # z = x_list[0]
        # for i in range(1, self.order):
        #     x_2 = x_list[i]
        #     z = z * x_2
        
        z = self.spec_convs(x)
        
        out = rearrange(z, 'B C H W D -> B H W D C')
        return out
    
class RMSNorm3D(nn.Module):
    def __init__(self, channels, name=None, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, 1, 1, channels))
        self.name = name
        self.eps = eps
        self.log_rms = True

    def forward(self, x):
        # x: [B, H, W, C]
        rms = torch.mean(x*x, dim=-1, keepdim=True) #(1, 2), keepdim=True) # dim 0 -1

    # if self.name and self.log_rms and wandb.run is not None:
    #     wandb.log({
    #         f"{self.name}/rms_mean": rms.mean().detach().cpu(),
    #         f"{self.name}/rms_std": rms.std().detach().cpu(),
    #         f"{self.name}/weight_mean": self.weight.mean().detach().cpu(),
    #     }, commit=False)
            
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
    def __init__(self, width, modes1, modes2, modes3, order, name1, name2, p_drop_MLP, drop_path, expansion_MLP):
        super().__init__()
        self.HO_conv = HO_Conv(width, width, modes1, modes2, modes3, order)
        self.norm1 = RMSNorm3D(width, name1) # nn.LayerNorm(width)
        self.norm2 = RMSNorm3D(width, name2)  # nn.LayerNorm(width)
        self.MLP = nn.Sequential(nn.Linear(width, width * expansion_MLP),
                                 nn.GELU(),
                                 nn.Dropout(p_drop_MLP),
                                 nn.Linear(width * expansion_MLP, width),
                                 nn.Dropout(p_drop_MLP))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.HO_conv(self.norm1(x)))
        x = x + self.drop_path(self.MLP(self.norm2(x)))
        return x
    
class HO_FNO_3D(nn.Module): 
        def __init__(self, width, depth, in_channels, out_channels, modes1, modes2, modes3, order, p_drop_MLP, drop_path_rate, expansion_MLP):
            super().__init__()
            self.D = 2
            self.depth = depth

            in_channels = in_channels

            self.linear_p = nn.Linear(in_channels, width)
            drop_rates = torch.linspace(0, drop_path_rate, depth)

            self.HO_blocks = nn.ModuleList([HO_block(width, modes1, modes2, modes3, order, name1=f"norm pre conv {l}", name2=f"norm pre mlp {l}", p_drop_MLP=p_drop_MLP, drop_path=drop_rates[l], expansion_MLP=expansion_MLP)
                                       for l in range(depth)])

            self.linear_q = nn.Linear(width, out_channels)
            
        def forward(self, x):
            # x: [B, H, W, C] (Channel last)
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
            #x = torch.cat([u, x_in], dim=-1)
            # Interesting: The point cloud is 1D before being projected into a 2D grid...

            x = self.linear_p(x) # lifting

            # Apply the Fourier layers sequentially
            for block in self.HO_blocks:
                x = block(x)

            x = self.linear_q(x)
            return x
        
        def get_grid(self, shape, device):
            batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
            gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
            gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
            return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
        
################################################################
# configs
################################################################
DATA_PATH = 'data/Plasticity/plas_N987_T20.mat'
#'../data/plasticity/buri_plas_N987_time.mat'

N = 987
ntrain = 900
ntest = 80

batch_size = 8
learning_rate = 0.001

epochs = int(os.environ.get("EPOCHS", "501"))
step_size = 100
gamma = 0.5

modes = 12
width = 32
out_dim = 4

s1 = 101
s2 = 31
t = 20

r1 = 1
r2 = 1
s1 = int(((s1 - 1) / r1) + 1)
s2 = int(((s2 - 1) / r2) + 1)

################################################################
# load data and data normalization
################################################################
reader = MatReader(DATA_PATH)
x_train = reader.read_field('input')[:ntrain, ::r1][:, :s1].reshape(ntrain,s1,1,1,1).repeat(1,1,s2,t,1)
y_train = reader.read_field('output')[:ntrain, ::r1, ::r2][:, :s1, :s2]
reader.load_file(DATA_PATH)
x_test = reader.read_field('input')[-ntest:, ::r1][:, :s1].reshape(ntest,s1,1,1,1).repeat(1,1,s2,t,1)
y_test = reader.read_field('output')[-ntest:, ::r1, ::r2][:, :s1, :s2]
print(x_train.shape, y_train.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

################################################################
# training and evaluation
################################################################
# model = FNO3d(modes, modes, 8, width).cuda()
model = HO_FNO_3D(width=64, depth=8, in_channels=4, out_channels=4, modes1=12, modes2=12, modes3=6, order=2, p_drop_MLP=0, drop_path_rate=0, expansion_MLP=4).cuda()
# model = torch.load('../model/plas_101'+str(500))
print(count_params(model))
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = LpLoss(size_average=False, p=2)


for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x) #.reshape(batch_size, s1, s2, t, out_dim)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()
        optimizer.step()

        train_l2 += loss.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x) #.reshape(batch_size, s1, s2, t, out_dim)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    train_reg /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(f"epoch: {ep} | train_l2: {train_l2}, train_reg: {train_reg} | test_l2: {test_l2} | time: {t2 - t1}")

    if ep%50==0:
        # torch.save(model, '../model/plas_101'+str(ep))

        truth = y[0].squeeze().detach().cpu().numpy()
        pred = out[0].squeeze().detach().cpu().numpy()
        ZERO = torch.zeros(s1,s2)
        truth_du = np.linalg.norm(truth[:,:,:,2:], axis=-1)
        pred_du = np.linalg.norm(pred[:,:,:,2:], axis=-1)

        lims = dict(cmap='RdBu_r', vmin=truth_du.min(), vmax=truth_du.max())
        fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
        t0,t1,t2,t3,t4 = 0,4,9,14,19
        ax[0,0].scatter(truth[:,:,0,0], truth[:,:,0,1], 10, truth_du[:,:,0],    **lims)
        ax[1,0].scatter(pred[:,:,0,0], pred[:,:,0,1],   10, pred_du[:,:,0],     **lims)
        ax[0,1].scatter(truth[:,:,4,0], truth[:,:,4,1], 10, truth_du[:,:,4],    **lims)
        ax[1,1].scatter(pred[:,:,4,0], pred[:,:,4,1],   10, pred_du[:,:,4],     **lims)
        ax[0,2].scatter(truth[:,:,9,0], truth[:,:,9,1], 10, truth_du[:,:,9],    **lims)
        ax[1,2].scatter(pred[:,:,9,0], pred[:,:,9,1],   10, pred_du[:,:,9],     **lims)
        ax[0,3].scatter(truth[:,:,14,0], truth[:,:,14,1],10, truth_du[:,:,14],  **lims)
        ax[1,3].scatter(pred[:,:,14,0], pred[:,:,14,1], 10, pred_du[:,:,14],    **lims)
        ax[0,4].scatter(truth[:,:,19,0], truth[:,:,19,1],10, truth_du[:,:,19],  **lims)
        ax[1,4].scatter(pred[:,:,19,0], pred[:,:,19,1], 10, pred_du[:,:,19],    **lims)
        fig.show()

# Original size of FNO: 18 883 748
# Depthwise: 599 204
