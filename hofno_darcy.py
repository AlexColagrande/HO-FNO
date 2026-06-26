import os
import argparse
import time
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from models.HO_FNO import HO_FNO
from utils.normalizer import UnitTransformer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb

parser = argparse.ArgumentParser('Training HO-FNO')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--slice_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='HO_FNO_Darcy')
parser.add_argument('--data_path', type=str, default='data')

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

args = parser.parse_args() # remove [] for sript

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat'
test_path = args.data_path + '/piececonst_r421_N1024_smooth2.mat'

ntrain = args.ntrain
ntest = 200
eval = args.eval
save_name = args.save_name

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def central_diff(x: torch.Tensor, h, resolution):
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    #x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = rearrange(x, 'b h w c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x,
              (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2 * h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y

r = args.downsample
h = int(((421 - 1) / r) + 1)
s = h
dx = 1.0 / s

train_data = scio.loadmat(train_path)
x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
#x_train = x_train.reshape(ntrain, -1)
x_train = torch.from_numpy(x_train).float()
y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
#y_train = y_train.reshape(ntrain, -1)
y_train = torch.from_numpy(y_train)

test_data = scio.loadmat(test_path)
x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
#x_test = x_test.reshape(ntest, -1)
x_test = torch.from_numpy(x_test).float()
y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
#y_test = y_test.reshape(ntest, -1)
y_test = torch.from_numpy(y_test)

x_normalizer = UnitTransformer(x_train)
y_normalizer = UnitTransformer(y_train)

x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)
y_train = y_normalizer.encode(y_train)

x_normalizer.cuda()
y_normalizer.cuda()

x = np.linspace(0, 1, s)
y = np.linspace(0, 1, s)
# x, y = np.meshgrid(x, y)
x, y = np.meshgrid(x, y, indexing='ij') # Added to have 2D 
# pos = np.c_[x.ravel(), y.ravel()]
pos = np.stack([x, y], axis=-1) # [h, h, 2]
pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

# pos_train = pos.repeat(ntrain, 1, 1)
pos_train = pos.repeat(ntrain, 1, 1, 1) # [B, h, h, 2]
#pos_test = pos.repeat(ntest, 1, 1)
pos_test = pos.repeat(ntest, 1, 1, 1) # [B, h, h, 2]
print("Dataloading is over.")

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                            batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                            batch_size=args.batch_size, shuffle=False)

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

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

print(args)
count_parameters(model)

if use_wandb:
        wandb.log({"n_params": count_parameters(model)}, step=0)

#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs, steps_per_epoch=len(train_loader))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
myloss = TestLoss(size_average=False)
de_x = TestLoss(size_average=False)
de_y = TestLoss(size_average=False) 
indices_n = torch.arange(s**2).reshape(1, 1, s**2, 1)  # Shape (1, 1, N, 1)
indices_m = torch.arange(args.slice_num).reshape(1, 1, 1, args.slice_num)  # Shape (1, 1, 1, M)
den= s // int(args.slice_num**0.5)

a = (indices_n // s) // den
b = (indices_n % s)// den  

target_index = a * int(args.slice_num**0.5) + b  # Shape (1, 1, N, 1)

mask = (indices_m == target_index).to(torch.float32)  # Shape (1, 1, N, M)
mask = mask.cuda()
my_l1 = TestLoss(p=1, size_average=False)

if eval:
    print("model evaluation")
    print(s, s)
    model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"), strict=False)
    model.eval()
    showcase = 10
    id = 0
    if not os.path.exists('./results/' + save_name + '/'):
        os.makedirs('./results/' + save_name + '/')

    with torch.no_grad():
        rel_err = 0.0
        with torch.no_grad():
            for x, fx, y in test_loader:
                id += 1
                x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()

                rel_err += tl

                if id < showcase:
                    print(id)
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(out[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                        "case_" + str(id) + "_pred.pdf"))
                    plt.close()
                    # ============ #
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(y[0, :].reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_gt.pdf"))
                    plt.close()
                    # ============ #
                    plt.figure()
                    plt.axis('off')
                    plt.imshow((y[0, :] - out[0, :]).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.0005, 0.0005)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_error.pdf"))
                    plt.close()
                    # ============ #
                    plt.figure()
                    plt.axis('off')
                    plt.imshow((fx[0, :].unsqueeze(-1)).reshape(85, 85).detach().cpu().numpy(), cmap='coolwarm')
                    plt.colorbar()
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/', "case_" + str(id) + "_input.pdf"))
                    plt.close()

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))


else:
    for ep in range(args.epochs):
        epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        train_loss = 0
        reg = 0
        for x, fx, y in train_loader:
            x, fx, y = x.cuda(), fx.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x, fx.unsqueeze(-1)) # B, N , 2, fx: B, N, y: B, N
            
            out = out.squeeze(-1) 
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            
            l2loss = myloss(out, y)

            # out = rearrange(out.unsqueeze(-1), 'b (h w) c -> b c h w', h=s)
            out = rearrange(out.unsqueeze(-1), 'b h w c -> b c h w', h=s)
            out = out[..., 1:-1, 1:-1].contiguous()
            out = F.pad(out, (1, 1, 1, 1), "constant", 0)
            #out = rearrange(out, 'b c h w -> b (h w) c')
            out = rearrange(out, 'b c h w -> b h w c')
            gt_grad_x, gt_grad_y = central_diff(y.unsqueeze(-1), dx, s)
            pred_grad_x, pred_grad_y = central_diff(out, dx, s)
            deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y, gt_grad_y)
            loss = l2loss + 0.1 * deriv_loss 
            loss.backward()

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += l2loss.item()
            reg += deriv_loss.item()
        scheduler.step()

        train_loss /= ntrain
        reg /= ntrain
        print(f"Epoch: {ep} | Reg : {reg:.5f} Train loss : {train_loss:.5f}")
        model.eval()
        rel_err = 0.0
        id = 0
        with torch.no_grad():
            for x, fx, y in test_loader:
                id += 1
                if id == 2:
                    vis = True
                else:
                    vis = False
                x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                out = model(x, fx=fx.unsqueeze(-1))
                out = out.squeeze(-1)
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl
        rel_err /= ntest
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_mem_gb = 0.0
        epoch_time = time.perf_counter() - epoch_start
        print("rel_err:{}".format(rel_err))
        print("Epoch {} Time: {:.2f}s Peak memory: {:.2f} GB".format(ep, epoch_time, peak_mem_gb))

        if use_wandb:
                wandb.log({
                    "epoch": ep,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/step_loss": train_loss,
                    "test/step_loss": rel_err,
                    "train/epoch_time_s": epoch_time,
                    "train/peak_memory_gb": peak_mem_gb,
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
