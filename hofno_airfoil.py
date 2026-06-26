import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from tqdm import *
from utils.testloss import TestLoss
from models.HO_FNO import HO_FNO
import wandb

parser = argparse.ArgumentParser('Training HO-FNO')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='HO_FNO_Airfoil')
parser.add_argument('--data_path', type=str, default='data/naca')

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
eval = args.eval
save_name = args.save_name
run_save_name = f"{save_name}_{time.strftime('%Y%m%d_%H%M%S')}"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    INPUT_X = args.data_path + '/NACA_Cylinder_X.npy'
    INPUT_Y = args.data_path + '/NACA_Cylinder_Y.npy'
    OUTPUT_Sigma = args.data_path + '/NACA_Cylinder_Q.npy'

    ntrain = 1000
    ntest = 200

    r1 = args.downsamplex
    r2 = args.downsampley
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 4]
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    # x_train = x_train.reshape(ntrain, -1, 2)
    # x_test = x_test.reshape(ntest, -1, 2)
    # y_train = y_train.reshape(ntrain, -1)
    # y_test = y_test.reshape(ntest, -1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, x_train, y_train),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, x_test, y_test),
                                              batch_size=args.batch_size,
                                              shuffle=False)

    print("Dataloading is over.")

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
    print(model)
    count_parameters(model)

    if use_wandb:
        wandb.log({"n_params": count_parameters(model)}, step=0)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    myloss = TestLoss(size_average=False)

    if eval:
        model.load_state_dict(torch.load("./checkpoints/" + save_name + ".pt"))
        model.eval()
        if not os.path.exists('./results/' + save_name + '/'):
            os.makedirs('./results/' + save_name + '/')

        rel_err = 0.0
        showcase = 10
        id = 0

        with torch.no_grad():
            for pos, fx, y in test_loader:
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                out = model(x, None).squeeze(-1)

                tl = myloss(out, y).item()
                rel_err += tl
                if id < showcase:
                    print(id)
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   x[0, :, 1].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   np.zeros([140, 35]),
                                   shading='auto',
                                   edgecolors='black', linewidths=0.1)
                    plt.colorbar()
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "input_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   x[0, :, 1].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   out[0, :].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1.2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "pred_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   x[0, :, 1].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   y[0, :].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(0, 1.2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "gt_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    plt.axis('off')
                    plt.pcolormesh(x[0, :, 0].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   x[0, :, 1].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   out[0, :].reshape(221, 51)[40:180, :35].detach().cpu().numpy() - \
                                   y[0, :].reshape(221, 51)[40:180, :35].detach().cpu().numpy(),
                                   shading='auto', cmap='coolwarm')
                    plt.colorbar()
                    plt.clim(-0.2, 0.2)
                    plt.savefig(
                        os.path.join('./results/' + save_name + '/',
                                     "error_" + str(id) + ".pdf"), bbox_inches='tight', pad_inches=0)
                    plt.close()

        rel_err /= ntest
        print("rel_err:{}".format(rel_err))
    else:
        for ep in range(args.epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            model.train()
            train_start = time.perf_counter()
            train_loss = 0

            for pos, fx, y in train_loader:

                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()  # x:B,N,2  fx:B,N,2  y:B,N
                optimizer.zero_grad()
                out = model(x, None).squeeze(-1)
                loss = myloss(out, y)
                loss.backward()

                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            train_time = time.perf_counter() - train_start

            train_loss = train_loss / ntrain
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            model.eval()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            rel_err = 0.0
            with torch.no_grad():
                for pos, fx, y in test_loader:
                    x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                    out = model(x, None).squeeze(-1)

                    tl = myloss(out, y).item()
                    rel_err += tl

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            else:
                peak_mem_gb = 0.0
            infer_time = time.perf_counter() - infer_start

            rel_err /= ntest
            print("rel_err:{}".format(rel_err))
            print("Epoch {} Train time: {:.2f}s | Inference time: {:.2f}s | Peak memory: {:.2f} GB".format(ep, train_time, infer_time, peak_mem_gb))
            if use_wandb:
                wandb.log({
                    "epoch": ep,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/step_loss": train_loss,
                    "test/step_loss": rel_err,
                    "train/epoch_time_s": train_time,
                    "inference/epoch_time_s": infer_time,
                    "system/peak_memory_gb": peak_mem_gb,
                }, step=ep)

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save model')
                torch.save(model.state_dict(), os.path.join('./checkpoints', run_save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('save model')
        final_ckpt_path = os.path.join('./checkpoints', run_save_name + '.pt')
        torch.save(model.state_dict(), final_ckpt_path)
        print(f"final checkpoint saved to: {final_ckpt_path}")


if __name__ == "__main__":
    main()
