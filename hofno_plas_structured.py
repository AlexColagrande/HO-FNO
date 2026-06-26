import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
from tqdm import *
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer
import wandb

parser = argparse.ArgumentParser('Training HO-FNO')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='HO_FNO_Plasticity')
parser.add_argument('--data_path', type=str, default='data/Plasticity/plas_N987_T20.mat')

# Arguments added for Wandb

parser.add_argument('--use_wandb', type=bool, default=True)
parser.add_argument('--wandb_project', type=str, default="HO-FNO")
parser.add_argument('--wandb_entity', type=str, default=None)  # optional
parser.add_argument('--wandb_name', type=str, default=None)    # optional run name
parser.add_argument('--wandb_tags', type=str, nargs='*', default=None)
parser.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'])

args = parser.parse_args()
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch 
from torch import nn 
from einops import rearrange 
from layers.spectral_convs import SpectralConv_2D_diag 
import wandb



class HO_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, order=2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.order = order

        self.proj = nn.Conv2d(in_channels, out_channels * order, kernel_size=1, bias=True)
        self.spec_convs = SpectralConv_2D_diag(in_channels, out_channels, modes1, modes2)
        #self.final_spec_conv = SpectralConv_2D_diag(in_channels, out_channels, modes1, modes2)
        

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
        rms = torch.mean(x*x, dim=-1, keepdim=True) #(1, 2), keepdim=True) # dim 0 -1
            
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

from model.Embedding import timestep_embedding

class HO_FNO(nn.Module): 
        def __init__(self, width, depth, in_channels, out_channels, modes1, modes2, order, p_drop_MLP, drop_path_rate, expansion_MLP):
            super().__init__()
            self.D = 2
            self.depth = depth
            self.width = width

            in_channels = in_channels

            self.linear_p = nn.Linear(in_channels, width)
            self.time_fc = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

            drop_rates = torch.linspace(0, drop_path_rate, depth)

            self.HO_blocks = nn.ModuleList([HO_block(width, modes1, modes2, order, name1=f"norm pre conv {l}", name2=f"norm pre mlp {l}", p_drop_MLP=p_drop_MLP, drop_path=drop_rates[l], expansion_MLP=expansion_MLP)
                                       for l in range(depth)])


            self.linear_q = nn.Linear(width, out_channels)
            
        def forward(self, x, fx, T):
            # x: [B, H, W, C] (Channel last)
            x = torch.cat([x, fx.unsqueeze(-1)], dim=-1)

            # if T is not None:
            # T is never None here
            # Time_emb = timestep_embedding(T, self.width).repeat(1, x.shape[1], 1)
            Time_emb = timestep_embedding(T, self.width)   # [B, width]
            Time_emb = Time_emb[:, :, None, :]          # [B, 1, 1, width]
            Time_emb = Time_emb.repeat(1, x.shape[1], x.shape[2], 1)  # [B, H, W, width]
            
            Time_emb = self.time_fc(Time_emb)
            x = self.linear_p(x) # lifting
            x = x + Time_emb

            # Apply the Fourier layers sequentially
            for block in self.HO_blocks:
                x = block(x)

            x = self.linear_q(x)
            return x

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def random_collate_fn(batch):
    shuffled_batch = []
    shuffled_u = None
    shuffled_t = None
    shuffled_a = None
    shuffled_pos = None
    for item in batch:
        pos = item[0]
        t = item[1]
        a = item[2]
        u = item[3]

        num_timesteps = t.size(0)
        permuted_indices = torch.randperm(num_timesteps)

        t = t[permuted_indices]
        u = u[..., permuted_indices]

        if shuffled_t is None:
            shuffled_pos = pos.unsqueeze(0)
            shuffled_t = t.unsqueeze(0)
            shuffled_u = u.unsqueeze(0)
            shuffled_a = a.unsqueeze(0)
        else:
            shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
            shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
            shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
            shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)

    shuffled_batch.append(shuffled_pos)
    shuffled_batch.append(shuffled_t)
    shuffled_batch.append(shuffled_a)
    shuffled_batch.append(shuffled_u)

    return shuffled_batch

DATA_PATH = args.data_path

N = 987
ntrain = 900
ntest = 80

s1 = 101
s2 = 31
T = 20
Deformation = 4

r1 = 1
r2 = 1
s1 = int(((s1 - 1) / r1) + 1)
s2 = int(((s2 - 1) / r2) + 1)

data = scio.loadmat(DATA_PATH)
input = torch.tensor(data['input'], dtype=torch.float)
output = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)
print(input.shape, output.shape)
x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2)
# x_train = x_train.reshape(ntrain, -1, 1)
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
# y_train = y_train.reshape(ntrain, -1, Deformation, T)
x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2)
# x_test = x_test.reshape(ntest, -1, 1)
y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
# y_test = y_test.reshape(ntest, -1, Deformation, T)
print(x_train.shape, y_train.shape)

x_normalizer = UnitTransformer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
x_normalizer.cuda()

x = np.linspace(0, 1, s1)
y = np.linspace(0, 1, s2)
# x, y = np.meshgrid(x, y)
x, y = np.meshgrid(x, y, indexing='ij') # Added to have 2D 
# pos = np.c_[x.ravel(), y.ravel()]
pos = np.stack([x, y], axis=-1) # [h, h, 2]
pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

# pos_train = pos.repeat(ntrain, 1, 1)
pos_train = pos.repeat(ntrain, 1, 1, 1) # [B, h, h, 2]
# pos_test = pos.repeat(ntest, 1, 1)
pos_test = pos.repeat(ntest, 1, 1, 1) # [B, h, h, 2]
print("Dataloading is over.")

t = np.linspace(0, 1, T)
t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
t_train = t.repeat(ntrain, 1)
t_test = t.repeat(ntest, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
                                            batch_size=args.batch_size, shuffle=True, collate_fn=random_collate_fn, num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test),
                                            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

print("Dataloading is over.")

# model = get_model(args).Model(space_dim=2,
#                                 n_hidden=args.n_hidden,
#                                 n_layers=args.n_layers,
#                                 Time_Input=True,
#                                 n_head=args.n_heads,
#                                 fun_dim=1,
#                                 out_dim=Deformation,
#                                 mlp_ratio=args.mlp_ratio,
#                                 slice_num=args.slice_num,
#                                 unified_pos=args.unified_pos,
#                                 H=s1,
#                                 W=s2).cuda()

model = HO_FNO(width=128, depth=8, in_channels=3, out_channels=4, modes1=24, modes2=12, order=2, p_drop_MLP=0, drop_path_rate=0.1, expansion_MLP=4).cuda()

use_wandb = bool(args.use_wandb) and (args.wandb_mode != 'disabled')
if use_wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        config=vars(args),  # logs all CLI args
    )

# Currently reported results with this config:
# model = HO_FNO(width=64, depth=8, in_channels=3, out_channels=4, modes1=64, modes2=12, order=2, p_drop_MLP=0, drop_path_rate=0, expansion_MLP=4).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

print(args)
print(model)
count_parameters(model)
if use_wandb:
        wandb.log({"n_params": count_parameters(model)}, step=0)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,  steps_per_epoch=len(train_loader))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.e-6)
myloss = TestLoss(size_average=False)

if eval:
    model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
    model.eval()
    if not os.path.exists('./results/' + save_name + '/'):
        os.makedirs('./results/' + save_name + '/')
    test_l2_step = 0
    test_l2_full = 0
    showcase = 10
    id = 0
    with torch.no_grad():
        for x, tim, fx, yy in test_loader:
            id += 1
            loss = 0
            x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
            bsz = x.shape[0]

            for t in range(T):
                y = yy[..., t:t + 1]
                input_T = tim[:, t:t + 1].reshape(bsz, 1)
                im = model(x, fx, T=input_T)
                loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                if t == 0:
                    pred = im.unsqueeze(-1)
                else:
                    pred = torch.cat((pred, im.unsqueeze(-1)), -1)

            if id < showcase:
                print(id)
                truth = y[0].reshape(101, 31, 4).squeeze().detach().cpu().numpy()
                pred_vis = im[0].reshape(101, 31, 4).squeeze().detach().cpu().numpy()
                truth_du = np.linalg.norm(truth[:, :, 2:], axis=-1)
                pred_du = np.linalg.norm(pred_vis[:, :, 2:], axis=-1)

                plt.axis('off')
                plt.scatter(truth[:, :, 0], truth[:, :, 1], 10, truth_du[:, :], cmap='coolwarm')
                plt.colorbar()
                plt.clim(0, 6)
                plt.savefig(
                    os.path.join('./results/' + save_name + '/',
                                    "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.axis('off')
                plt.scatter(pred_vis[:, :, 0], pred_vis[:, :, 1], 10, pred_du[:, :], cmap='coolwarm')
                plt.colorbar()
                plt.clim(0, 6)
                plt.savefig(
                    os.path.join('./results/' + save_name + '/',
                                    "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.axis('off')
                plt.scatter(truth[:, :, 0], truth[:, :, 1], 10, pred_du[:, :] - truth_du[:, :], cmap='coolwarm')
                plt.colorbar()
                plt.clim(-0.2, 0.2)
                plt.savefig(
                    os.path.join('./results/' + save_name + '/',
                                    "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                plt.close()

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

    print("test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(test_l2_step / ntest / T, test_l2_full / ntest))
else:
    for ep in range(args.epochs):

        model.train()
        train_l2_step = 0

        for x, tim, fx, yy in train_loader:
            x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
            bsz = x.shape[0]

            for t in range(T):
                y = yy[..., t:t + 1]
                input_T = tim[:, t:t + 1].reshape(bsz, 1)  # B,step
                im = model(x, fx, T=input_T)

                loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                train_l2_step += loss.item()
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        scheduler.step()

        model.eval()
        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for x, tim, fx, yy in test_loader:
                loss = 0
                x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
                bsz = x.shape[0]

                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    im = model(x, fx, T=input_T)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im.unsqueeze(-1)
                    else:
                        pred = torch.cat((pred, im.unsqueeze(-1)), -1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()

        print("Epoch {} , train_step_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(ep,
                                                                                                            train_l2_step / ntrain / T,
                                                                                                            test_l2_step / ntest / T,
                                                                                                            test_l2_full / ntest))
        if use_wandb:
                wandb.log({
                    "epoch": ep,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/step_loss": train_l2_step / ntrain / T,
                    "test/step_loss": test_l2_step / ntest / T,
                    "test/full_loss": test_l2_full / ntest,
                }, step=ep)

        if ep % 100 == 0:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            print('save model')
            torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    print('save model')
    torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))
