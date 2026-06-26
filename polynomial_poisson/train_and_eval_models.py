from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.HO_FNO import HO_FNO
from models.FNO import FNO


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    denominator = torch.norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-8)
    return (numerator / denominator).mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    rel_sum = 0.0
    count = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            batch_size = xb.shape[0]
            mse_sum += float(nn.functional.mse_loss(pred, yb, reduction="sum").item())
            rel_sum += float(relative_l2(pred, yb).item()) * batch_size
            count += batch_size
    pixels = count * loader.dataset.tensors[1].shape[1] * loader.dataset.tensors[1].shape[2] * loader.dataset.tensors[1].shape[3]
    return {
        "mse": mse_sum / pixels,
        "rel_l2": rel_sum / count,
    }


def build_model(
    model_name: str,
    input_channels: int,
    output_channels: int,
    modes: int,
    depth: int,
    order: int,
) -> nn.Module:
    shared_kwargs = dict(
        width=64,
        depth=depth,
        in_channels=input_channels + 2,
        out_channels=output_channels,
        modes1=modes,
        modes2=modes,
        p_drop_MLP=0.0,
        drop_path_rate=0.0,
        expansion_MLP=2,
        two_thirds_zero_padding=False,
    )
    if model_name == "HO-FNO":
        return HO_FNO(**shared_kwargs, order=order)
    if model_name == "FNO":
        return FNO(**shared_kwargs)
    raise ValueError(f"Unknown model {model_name}")


def train_one_model(
    model_name: str,
    dataset: dict,
    run_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    depth: int,
    order: int,
    wandb_enabled: bool,
    wandb_project: str,
) -> dict:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    metadata = dataset["metadata"]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = build_model(
        model_name,
        x_train.shape[-1],
        y_train.shape[-1],
        metadata["modes"],
        depth,
        order,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model_dir = run_dir / model_name.replace("-", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    history_path = model_dir / "history.csv"
    wandb_run = None
    if wandb_enabled:
        wandb_run = wandb.init(
            project=wandb_project,
            name=f"{metadata['name']}_seed{seed}_order{order}_depth{depth}_{model_name}",
            dir=str(model_dir),
            config={
                "dataset": metadata["name"],
                "dataset_description": metadata["description"],
                "model": model_name,
                "seed": seed,
                "depth": depth,
                "order": order,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "modes": metadata["modes"],
            },
            reinit=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

    best_train_mse = math.inf
    started_at = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "train_rel_l2"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            train_mse_sum = 0.0
            train_rel_sum = 0.0
            train_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                batch_size_now = xb.shape[0]
                train_mse_sum += float(nn.functional.mse_loss(pred.detach(), yb, reduction="sum").item())
                train_rel_sum += float(relative_l2(pred.detach(), yb).item()) * batch_size_now
                train_count += batch_size_now

            pixels = train_count * y_train.shape[1] * y_train.shape[2] * y_train.shape[3]
            train_mse = train_mse_sum / pixels
            train_rel = train_rel_sum / train_count
            best_train_mse = min(best_train_mse, train_mse)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_mse": f"{train_mse:.8f}",
                    "train_rel_l2": f"{train_rel:.8f}",
                }
            )
            if epoch <= 3 or epoch % 10 == 0 or epoch == epochs:
                print(
                    f"[{metadata['name']}] {model_name} epoch {epoch:03d}/{epochs} "
                    f"train_mse={train_mse:.6f} train_rel_l2={train_rel:.6f}",
                    flush=True,
                )
            if wandb_run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/mse": train_mse,
                        "train/rel_l2": train_rel,
                    }
                )

    test_metrics = evaluate(model, test_loader, device)
    weights_path = model_dir / "final_weights.pt"
    torch.save(
        {
            "dataset": metadata["name"],
            "model": model_name,
            "depth": depth,
            "order": order,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        weights_path,
    )
    if wandb_run is not None:
        wandb.log(
            {
                "epoch": epochs,
                "test/mse": test_metrics["mse"],
                "test/rel_l2": test_metrics["rel_l2"],
                "model/n_params": float(sum(p.numel() for p in model.parameters())),
            }
        )
        wandb.finish()
    wall_time_min = (time.time() - started_at) / 60.0
    summary = {
        "dataset": metadata["name"],
        "dataset_description": metadata["description"],
        "model": model_name,
        "seed": seed,
        "modes": metadata["modes"],
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "depth": depth,
        "order": order,
        "params": sum(p.numel() for p in model.parameters()),
        "best_train_mse": best_train_mse,
        "test_mse": test_metrics["mse"],
        "test_rel_l2": test_metrics["rel_l2"],
        "wall_time_min": wall_time_min,
        "history_csv": str(history_path),
        "final_weights": str(weights_path),
    }
    with (model_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset.pt")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="polynomial_poisson")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = dataset["metadata"]
    run_dir = (
        Path(args.output_root)
        / metadata["name"]
        / f"order_{args.order}"
        / f"depth_{args.depth}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["HO-FNO"]
    summaries = []
    for idx, model_name in enumerate(model_names):
        summaries.append(
            train_one_model(
                model_name=model_name,
                dataset=dataset,
                run_dir=run_dir,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed + idx,
                depth=args.depth,
                order=args.order,
                wandb_enabled=args.wandb,
                wandb_project=args.wandb_project,
            )
        )

    comparison_path = run_dir / "comparison.json"
    if comparison_path.is_file():
        with comparison_path.open("r", encoding="utf-8") as handle:
            comparison = json.load(handle)
    else:
        comparison = {
            "dataset": metadata["name"],
            "description": metadata["description"],
            "creation_script": metadata["creation_script"],
            "results": [],
        }

    existing = {entry["model"]: entry for entry in comparison["results"]}
    for summary in summaries:
        existing[summary["model"]] = summary
    comparison["results"] = [existing["HO-FNO"]] if "HO-FNO" in existing else []

    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
