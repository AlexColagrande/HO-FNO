import os

# GPU selection must happen before importing torch.
# Priority:
# 1) If CUDA_VISIBLE_DEVICES is already set, keep it.
# 2) Else, if GPU_ID is set (e.g. GPU_ID=1), map it to CUDA_VISIBLE_DEVICES.
if "CUDA_VISIBLE_DEVICES" not in os.environ and "GPU_ID" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_ID"]

import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities import *
from torch.optim import AdamW

try:
    import wandb
except ImportError:
    wandb = None

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(0)

class IPHI(nn.Module):
    # Li defaults: width = 32 
    def __init__(self, width=32):
        super().__init__()
        """
        inverse phi: x -> xi
        """
        self.width = width # it's used also in the forward
        self.fc0 = nn.Linear(4, width) # 4 is: x, y coordinates and the 2 polar coordinates.
        self.fc_code = nn.Linear(42, width)
        self.fc_no_code = nn.Linear(3*width, 4*width)
        self.fc1 = nn.Linear(4*width, 4*width)
        self.fc2 = nn.Linear(4*width, 4*width)
        self.fc3 = nn.Linear(4*width, 2)
        self.center = torch.tensor([0.5,0.5], device="cuda").reshape(1,1,2)

        self.B = torch.pi*torch.pow(2, torch.arange(0, width//4, dtype=torch.float, device="cuda")).reshape(1,1,1,width//4)

    def forward(self, x, code=None):
        # x: [batch, N_grid, 2]
        # code: [batch, N_features]

        # some feature engineering
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.width//4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.width)

        if code!= None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1,xd.shape[1],1)
            xd = torch.cat([cd,xd],dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = F.gelu(xd)
        xd = self.fc2(xd)
        xd = F.gelu(xd)
        xd = self.fc3(xd)
        return x + x * xd
    
class SpectralConv2d_diag(nn.Module):
    def __init__(self, channels, modes1, modes2, s1=32, s2=32):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2
        self.channels = channels

        self.scale = (1 / (channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, channels, x,y ), (channels, x,y) -> (batch, channels, x,y)
        return torch.einsum("bcxy,cxy->bcxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        # Channel-first
        
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * torch.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * torch.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y

import torch 
from torch import nn 
from einops import rearrange 
from layers.spectral_convs import SpectralConv_2D_diag

class HO_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, order=2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.order = order

        #self.proj = nn.Conv2d(in_channels, out_channels * order, kernel_size=1, bias=True)
        self.spec_convs = SpectralConv_2D_diag(in_channels, out_channels, modes1, modes2)
        #self.final_spec_conv = SpectralConv_2D_diag(in_channels, out_channels, modes1, modes2)
        

    def forward(self, x):
        # x: [B, H, W, C]
        x = rearrange(x, 'B H W C -> B C H W')

        #x = self.proj(x)

        # x_list = torch.chunk(x, self.order, dim=1)
        # z = x_list[0]
        # for i in range(1, self.order):
        #     x_2 = x_list[i]
        #     z = z * x_2
        
        z = self.spec_convs(x)
        
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
    def __init__(self, width, modes1, modes2, order, name1, name2, p_drop_MLP, drop_path, expansion_MLP):
        super().__init__()
        self.HO_conv = HO_Conv(width, width, modes1, modes2, order)
        self.norm1 = RMSNorm2D(width, name1) # nn.LayerNorm(width)
        self.norm2 = RMSNorm2D(width, name2)  # nn.LayerNorm(width)
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
    
class HO_FNO(nn.Module): 
        def __init__(self, width, depth, in_channels, out_channels, modes1, modes2, order, p_drop_MLP, drop_path_rate, expansion_MLP, s1, s2):
            super().__init__()
            self.D = 2
            self.depth = depth

            in_channels = in_channels

            self.linear_p = nn.Linear(in_channels, width)
            drop_rates = torch.linspace(0, drop_path_rate, depth)

            # GEO-FNO Encoder
            self.geo_enc = SpectralConv2d_diag(width, modes1, modes2, s1, s2)

            self.HO_blocks = nn.ModuleList([HO_block(width, modes1, modes2, order, name1=f"norm pre conv {l}", name2=f"norm pre mlp {l}", p_drop_MLP=p_drop_MLP, drop_path=drop_rates[l], expansion_MLP=expansion_MLP)
                                       for l in range(depth)])
            
            self.geo_dec = SpectralConv2d_diag(width, modes1, modes2, s1, s2)


            self.linear_q = nn.Linear(width, out_channels)
            
        def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
            # x: [B, H, W, C] (Channel last)
            x = u # Here the grid is the input
            #x = torch.cat([u, x_in], dim=-1)
            # Interesting: The point cloud is 1D before being projected into a 2D grid...

            x = self.linear_p(x) # lifting
           
            # Geo encoder
            x = rearrange(x, 'B N C -> B C N')
            x = self.geo_enc(x, x_in=u, iphi=iphi, code=code)
            x = rearrange(x, 'B C ... -> B ... C')
            x = F.gelu(x)
            
            #x = rearrange(x, 'B C H W -> B H W C')

            # Apply the Fourier layers sequentially
            for block in self.HO_blocks:
                x = block(x)

            # Geo decoder
            x = rearrange(x, 'B ... C -> B C ...')
            x = self.geo_dec(x, x_out=u, iphi=iphi, code=code)
            x = rearrange(x, 'B C ... -> B ... C')
            x = self.linear_q(x)
            return x

################################################################
# configs
################################################################
Ntotal = 2000
ntrain = 1000
ntest = 200

batch_size = 1
learning_rate_fno = 0.001
learning_rate_iphi = 0.0005

epochs = int(os.environ.get("EPOCHS", "501"))

modes1, modes2 = 20, 10
s1, s2 = 40, 40
width = 128
depth = 8
drop_path_rate=0.1

# W&B (optional)
# Set USE_WANDB=1 to enable tracking.
use_wandb = os.getenv("USE_WANDB", "0") == "1"
wandb_project = os.getenv("WANDB_PROJECT", "HO-FNO")
wandb_entity = os.getenv("WANDB_ENTITY", None)
wandb_run_name = os.getenv("WANDB_RUN_NAME", "hofno_elas")

################################################################
# load data and data normalization
################################################################
PATH_Sigma = './data/Meshes/Random_UnitCell_sigma_10.npy'
PATH_XY = './data/Meshes/Random_UnitCell_XY_10.npy'
PATH_rr = './data/Meshes/Random_UnitCell_rr_10.npy'

input_rr = np.load(PATH_rr)
input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1,0)
input_s = np.load(PATH_Sigma)
input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)
input_xy = np.load(PATH_XY)
input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)

train_rr = input_rr[:ntrain]
test_rr = input_rr[-ntest:]
train_s = input_s[:ntrain]
test_s = input_s[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]

# from utils.normalizer import UnitTransformer
# y_normalizer = UnitTransformer(train_s)

# train_s = y_normalizer.encode(train_s)
# y_normalizer.cuda()


print(train_rr.shape, train_s.shape, train_xy.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_rr, train_s, train_xy), batch_size=batch_size, shuffle=True) 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_rr, test_s, test_xy), batch_size=batch_size, shuffle=False) 

################################################################
# training and evaluation
################################################################
model = HO_FNO(width, depth=depth, in_channels=2, out_channels=1, modes1=modes1, modes2=modes2, order=2, p_drop_MLP=0, drop_path_rate=drop_path_rate, expansion_MLP=4, s1=s1, s2=s2).cuda()
#model = FNO2d(modes1, modes2, width, depth, in_channels=2, out_channels=1, s1=s1, s2=s2).cuda()
model_iphi = IPHI().cuda()
print(f"parmeters: mail: {count_params(model)} | IPHI: {count_params(model_iphi)} | Total: {count_params(model) + count_params(model_iphi)}")

optimizer_fno = AdamW(model.parameters(), lr=learning_rate_fno, weight_decay=1e-4)
scheduler_fno = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fno, T_max = 501)
optimizer_iphi = AdamW(model_iphi.parameters(), lr=learning_rate_iphi, weight_decay=1e-4)
scheduler_iphi = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_iphi, T_max = 501)

wandb_run = None
if use_wandb:
    if wandb is None:
        print("[W&B] USE_WANDB=1 but wandb is not installed. Continuing without W&B.")
    else:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={
                "ntrain": ntrain,
                "ntest": ntest,
                "batch_size": batch_size,
                "learning_rate_fno": learning_rate_fno,
                "learning_rate_iphi": learning_rate_iphi,
                "epochs": epochs,
                "modes1": modes1,
                "modes2": modes2,
                "s1": s1,
                "s2": s2,
                "width": width,
                "depth": depth,
                "drop_path_rate": drop_path_rate,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "model": "HO-FNO + IPHI",
            },
        )
        wandb.watch(model, log="gradients", log_freq=50)

myloss = LpLoss(size_average=False)
N_sample = 1000
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for rr, sigma, mesh in train_loader:
        rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()

        optimizer_fno.zero_grad()
        optimizer_iphi.zero_grad()
        out = model(mesh, code=rr, iphi=model_iphi)
        # out = y_normalizer.decode(out)
        # sigma = y_normalizer.decode(sigma)

        loss = myloss(out.reshape(batch_size, -1), sigma.reshape(batch_size, -1))
        loss.backward()

        optimizer_fno.step()
        optimizer_iphi.step()
        train_l2 += loss.item()

    scheduler_fno.step()
    scheduler_iphi.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for rr, sigma, mesh in test_loader:
            rr, sigma, mesh = rr.cuda(), sigma.cuda(), mesh.cuda()
            out = model(mesh, x_in=mesh, code=rr, iphi=model_iphi)
            # out = y_normalizer.decode(out)
            test_l2 += myloss(out.view(batch_size, -1), sigma.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(f"Epoch {ep} | train_l2: {train_l2:.5f} | test_l2: {test_l2:.5f} | Time: {t2 - t1:.2f} s")

    if wandb_run is not None:
        wandb.log({
            "epoch": ep,
            "train_l2": train_l2,
            "test_l2": test_l2,
            "lr_fno": optimizer_fno.param_groups[0]["lr"],
            "lr_iphi": optimizer_iphi.param_groups[0]["lr"],
            "epoch_time_s": t2 - t1,
        })

    if ep%100==0:
        os.makedirs("../model", exist_ok=True)
        torch.save(model, '../model/elas_v2_'+str(ep))
        torch.save(model_iphi, '../model/elas_v2_iphi_'+str(ep))
        XY = mesh[-1].squeeze().detach().cpu().numpy()
        truth = sigma[-1].squeeze().detach().cpu().numpy()
        pred = out[-1].squeeze().detach().cpu().numpy()

        lims = dict(cmap='RdBu_r', vmin=truth.min(), vmax=truth.max())
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        ax[0].scatter(XY[:, 0], XY[:, 1], 100, truth, edgecolor='w', lw=0.1, **lims)
        ax[1].scatter(XY[:, 0], XY[:, 1], 100, pred, edgecolor='w', lw=0.1, **lims)
        ax[2].scatter(XY[:, 0], XY[:, 1], 100, truth - pred, edgecolor='w', lw=0.1, **lims)
        fig.show()
        plt.savefig('output.png')

        if wandb_run is not None:
            wandb.log({"qualitative/output": wandb.Image('output.png')}, step=ep)

if wandb_run is not None:
    wandb.finish()
