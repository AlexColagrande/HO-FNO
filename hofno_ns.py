"""
The following code is the code used for Higher Order FNO in Table 2 for the Navier-Stokes 
Dataset (Our best result). To ensure complete comparability with the literature we build on the code of LaMO and we did not change anything beyond the hyperparameters of Optimizer and Scheduler, we had do remove the reshape from the dataloaders since LaMO works on flatten inputs while HO-FNO maintains the input at the dimension of the problem. Any modification is done after leaving commented the removed part of code present in the official LaMO codebase for maximum transparency.
"""

import torch 
from torch import nn 
from einops import rearrange 
from layers.spectral_convs import SpectralConv_2D_diag 
import numpy as np

class SpectralConv_2D_diag(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        assert in_channels == out_channels, "Channel-diagonal spectral conv requires in_channels == out_channels"
        self.modes1, self.modes2 = modes1, modes2
        self.out_channels = out_channels

        self.scale = (1 / (in_channels))

        self.weights1 = nn.Parameter(self.scale * torch.randn((in_channels, modes1, modes2), dtype=torch.cfloat)) 
        self.weights2 = nn.Parameter(self.scale * torch.randn((in_channels, modes1, modes2), dtype=torch.cfloat))

    # Complex multiplication 2d
    def compl_mul2d(self, input, weights):
        # (batch, channels, x,y ), (channels, out_channel, x,y) -> (batch, channels, x,y)
        return torch.einsum("B C H W, C H W-> B C H W", input, weights)
    
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x = x.to(torch.float32)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1), dtype=torch.cfloat, device=x.device)

        tmp1 = torch.zeros_like(out_ft)
        tmp2 = torch.zeros_like(out_ft)

        tmp1[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        tmp2[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        out_ft = tmp1 + tmp2

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x

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
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, 1, channels))
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
    def __init__(self, width, modes1, modes2, order, p_drop_MLP, drop_path, expansion_MLP, two_thirds_zero_padding=False):
        super().__init__()
        self.HO_conv = HO_Conv(width, width, modes1, modes2, order)
        self.norm1 = RMSNorm2D(width) 
        self.norm2 = RMSNorm2D(width) 
        self.MLP = nn.Sequential(nn.Linear(width, width * expansion_MLP),
                                 nn.GELU(),
                                 nn.Dropout(p_drop_MLP),
                                 nn.Linear(width * expansion_MLP, width),
                                 nn.Dropout(p_drop_MLP))
        self.drop_path = DropPath(drop_path)
        self.two_thirds_zero_padding = two_thirds_zero_padding

    def _pad_2_3_rule(self, x):
        orig_h, orig_w = x.shape[1], x.shape[2]
        pad_h = int((2 * orig_h) / 3)
        pad_w = int((2 * orig_w) / 3)
        if pad_h == 0 and pad_w == 0:
            return x, orig_h, orig_w
        x = rearrange(x, 'B H W C -> B C H W')
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        x = rearrange(x, 'B C H W -> B H W C')
        return x, orig_h, orig_w

    def _unpad_2_3_rule(self, x, orig_h, orig_w):
        return x[:, :orig_h, :orig_w, :]

    def forward(self, x):
        x_conv = self.norm1(x)
        if self.two_thirds_zero_padding:
            x_conv, orig_h, orig_w = self._pad_2_3_rule(x_conv)
            x_conv = self.HO_conv(x_conv)
            x_conv = self._unpad_2_3_rule(x_conv, orig_h, orig_w)
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

        in_channels = in_channels

        self.linear_p = nn.Linear(in_channels, width)
        drop_rates = torch.linspace(0, drop_path_rate, depth)

        self.HO_blocks = nn.ModuleList([HO_block(width, modes1, modes2, order, p_drop_MLP=p_drop_MLP, drop_path=drop_rates[l], expansion_MLP=expansion_MLP, two_thirds_zero_padding=two_thirds_zero_padding)
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

###################################################################################
# Original LaMO training code
###################################################################################

import os
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import numpy as np
import torch
from tqdm import *
from utils.testloss import TestLoss

import h5py
import time
import wandb

parser = argparse.ArgumentParser('Training HO-FNO')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='HO_FNO_NavierStokes')
parser.add_argument('--data_path', type=str, default='data/NavierStokes_V1e-5_N1200_T20.mat')

# Arguments added for HO-FNO
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--depth', type=int, required=True)
parser.add_argument('--in_channels', type=int, required=True)
parser.add_argument('--out_channels', type=int, required=True)
parser.add_argument('--modes1', type=int, required=True)
parser.add_argument('--modes2', type=int, required=True)
parser.add_argument('--order', type=int, required=True)
parser.add_argument('--p_drop_MLP', type=float, required=True)
parser.add_argument('--drop_path_rate', type=float, required=True)
parser.add_argument('--eta_min', type=float, required=True)
parser.add_argument('--expansion_MLP', type=int, required=True)
parser.add_argument('--2_3_zero_padding', dest='two_thirds_zero_padding', action='store_true')

parser.add_argument('--use_wandb', type=bool, default=True)
parser.add_argument('--wandb_project', type=str, default="HO-FNO")
parser.add_argument('--wandb_entity', type=str, default=None)  # optional
parser.add_argument('--wandb_name', type=str, default=None)    # optional run name
parser.add_argument('--wandb_tags', type=str, nargs='*', default=None)
parser.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'])

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


###################################################################################
# Data parameters from the original LaMO codebase
###################################################################################

data_path = args.data_path
ntrain = 1000
ntest = 200
T_in = 10
T = 10
step = 1
eval = args.eval
save_name = args.save_name
dated_save_name = f"{save_name}_{time.strftime('%Y%m%d')}"

###################################################################################
# Utilities from the original LaMO codebase
###################################################################################

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def load_v73_mat_file(file_path):
    """Load a MATLAB v7.3 .mat file into a Python dictionary."""
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        def recursive_load(h5_obj):
            if isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]
            elif isinstance(h5_obj, h5py.Group):
                return {key: recursive_load(h5_obj[key]) for key in h5_obj.keys()}
            else:
                return None
        
        for key in f.keys():
            data_dict[key] = recursive_load(f[key])
    
    return data_dict

###################################################################################
# Data processing and dataloaders (Without modifications except commenting the 
# reshape to keep the data at the original dimension)
###################################################################################

def main():
    r = args.downsample
    h = int(((64 - 1) / r) + 1)

    data = scio.loadmat(data_path)

    print(data['u'].shape)

    train_a = data['u'][:ntrain, ::r, ::r, :T_in][:, :h, :h, :]
    #train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
    train_a = torch.from_numpy(train_a)
    train_u = data['u'][:ntrain, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    #train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
    train_u = torch.from_numpy(train_u)
            
    test_a = data['u'][-ntest:, ::r, ::r, :T_in][:, :h, :h, :]
    #test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
    test_a = torch.from_numpy(test_a)
    test_u = data['u'][-ntest:, ::r, ::r, T_in:T + T_in][:, :h, :h, :]
    #test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
    test_u = torch.from_numpy(test_u)

    x = np.linspace(0, 1, h)
    y = np.linspace(0, 1, h)
    # x, y = np.meshgrid(x, y)
    x, y = np.meshgrid(x, y, indexing='ij') # Added to have 2D mesh to stack to the input
    # pos = np.c_[x.ravel(), y.ravel()]
    pos = np.stack([x, y], axis=-1) # [h, h, 2]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    # pos_train = pos.repeat(ntrain, 1, 1) # Changed to keep the data 2-dimensional
    pos_train = pos.repeat(ntrain, 1, 1, 1) # [B, h, h, 2]
    # pos_test = pos.repeat(ntest, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1, 1) # [B, h, h, 2]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_u),
                                                batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_u),
                                                batch_size=args.batch_size, shuffle=False)
    print("Dataloading is over.")

    use_wandb = bool(args.use_wandb) and (args.wandb_mode != 'disabled')
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            config=vars(args),
        )
    
    
    model = HO_FNO(width=args.width,
                        depth=args.depth,
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        modes1=args.modes1,
                        modes2=args.modes2,
                        order=args.order,
                        p_drop_MLP=args.p_drop_MLP,
                        drop_path_rate=args.drop_path_rate,
                        expansion_MLP=args.expansion_MLP,
                        two_thirds_zero_padding=args.two_thirds_zero_padding).cuda()

   
###################################################################################
# Optimizer and Schedueler (Without modifications except for hyperparameters)
###################################################################################
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    print(args)
    print(model)
    n_params = count_parameters(model)

    if use_wandb:
        wandb.log({"n_params": n_params}, step=0)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    myloss = TestLoss(size_average=False)

###################################################################################
# Training loop from the original LaMO codebase
###################################################################################

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
        model.eval()
        showcase = 10
        id = 0

        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        test_l2_full = 0
        with torch.no_grad():
            for x, fx, yy in test_loader:
                id += 1
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                bsz = x.shape[0]
                for t in range(0, T, step):
                    im = model(x, fx=fx)

                    fx = torch.cat((fx[..., step:], im), dim=-1)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                if id < showcase:
                    print(id)
                    plt.figure()
                    plt.axis('off')
                    # plt.imshow(im[0, :, 0].reshape(64, 64).detach().cpu().numpy(), cmap='coolwarm')
                    plt.imshow(im[0, :, 0].detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-3, 3)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                        "case_" + str(id) + "_pred_" + str(20) + ".pdf"))
                    plt.close()
                    # ============ #
                    plt.figure()
                    plt.axis('off')
                    # plt.imshow(yy[0, :, t].reshape(64, 64).detach().cpu().numpy(), cmap='coolwarm')
                    plt.imshow(yy[0, :, t].detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-3, 3)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_gt_" + str(20) + ".pdf"))
                    plt.close()
                    # ============ #
                    plt.figure()
                    plt.axis('off')
                    # plt.imshow((im[0, :, 0].reshape(64, 64) - yy[0, :, t].reshape(64, 64)).detach().cpu().numpy(),
                    #             cmap='coolwarm')
                    plt.imshow((im[0, :, 0] - yy[0, :, t]).detach().cpu().numpy(),
                                cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-2, 2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_error_" + str(20) + ".pdf"))
                    plt.close()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            print(test_l2_full / ntest)

    else:
        for ep in range(args.epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            model.train()
            train_start = time.perf_counter()
            train_l2_step = 0
            train_l2_full = 0
            for x, fx, yy in train_loader:
                loss = 0
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x: B,4096,2    fx: B,4096,T   y: B,4096,T
                bsz = x.shape[0]

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(x, fx=fx).unsqueeze(-1)  # B , 4096 , 1
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    fx = torch.cat((fx[..., step:], y), dim=-1)  # detach() & groundtruth

                train_l2_step += loss.item()
                train_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            train_time = time.perf_counter() - train_start
            
            
            test_l2_step = 0
            test_l2_full = 0

            model.eval()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_start = time.perf_counter()

            with torch.no_grad():
                for x, fx, yy in test_loader:
                    loss = 0
                    x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()  # x : B, 4096, 2  fx : B, 4096  y : B, 4096, T
                    bsz = x.shape[0]
                    for t in range(0, T, step):
                        y = yy[..., t:t + step]
                        im = model(x, fx=fx) #.unsqueeze(-1)
                        loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)
                        fx = torch.cat((fx[..., step:], im), dim=-1)

                    test_l2_step += loss.item()
                    test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time = time.perf_counter() - infer_start
            

            print(
                "Epoch {} , train_step_loss:{:.5f} , train_full_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(
                    ep, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
                        test_l2_full / ntest))
            if torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            else:
                peak_mem_gb = 0.0
            print("Epoch {} Train time: {:.2f}s | Inference time: {:.2f}s | Peak memory: {:.2f} GB".format(ep, train_time, infer_time, peak_mem_gb))

            if use_wandb:
                wandb.log({
                    "epoch": ep,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/step_loss": train_l2_step / ntrain / (T / step),
                    "train/full_loss": train_l2_full / ntrain,
                    "test/step_loss": test_l2_step / ntest / (T / step),
                    "test/full_loss": test_l2_full / ntest,
                    "train/epoch_time_s": train_time,
                    "inference/epoch_time_s": infer_time,
                    "system/peak_memory_gb": peak_mem_gb,
                }, step=ep)
            

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', dated_save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        final_ckpt_path = os.path.join('./checkpoints', dated_save_name + '.pt')
        torch.save(model.state_dict(), final_ckpt_path)
        print(f"final checkpoint saved to: {final_ckpt_path}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
