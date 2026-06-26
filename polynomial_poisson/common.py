from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


RESOLUTION = 64
TRAIN_SAMPLES = 500
TEST_SAMPLES = 200


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    modes: int


def make_grid(resolution: int = RESOLUTION) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(0.0, 1.0, resolution)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    return x, y


GRID_X, GRID_Y = make_grid()


def normalize_per_sample(field: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = field.mean(dim=(-2, -1), keepdim=True)
    std = field.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (field - mean) / std


def sample_wave_field(
    generator: torch.Generator,
    resolution: int,
    min_freq: float,
    max_freq: float,
    n_waves: int,
    anisotropy_x: float = 1.0,
    anisotropy_y: float = 1.0,
) -> torch.Tensor:
    x = GRID_X[:resolution, :resolution]
    y = GRID_Y[:resolution, :resolution]
    field = torch.zeros((resolution, resolution), dtype=torch.float32)
    for _ in range(n_waves):
        freq = (min_freq + (max_freq - min_freq) * torch.rand(1, generator=generator)).item()
        theta = (2.0 * math.pi * torch.rand(1, generator=generator)).item()
        phase_1 = (2.0 * math.pi * torch.rand(1, generator=generator)).item()
        phase_2 = (2.0 * math.pi * torch.rand(1, generator=generator)).item()
        amp_1 = torch.randn(1, generator=generator).item()
        amp_2 = torch.randn(1, generator=generator).item()
        kx = anisotropy_x * freq * math.cos(theta)
        ky = anisotropy_y * freq * math.sin(theta)
        phase = 2.0 * math.pi * (kx * x + ky * y)
        field += amp_1 * torch.sin(phase + phase_1)
        field += 0.5 * amp_2 * torch.cos(phase + phase_2)
    return normalize_per_sample(field.unsqueeze(0)).squeeze(0)


def poisson_filter(field: torch.Tensor, alpha_x: float, alpha_y: float) -> torch.Tensor:
    resolution = field.shape[-1]
    freq_x = torch.fft.fftfreq(resolution, d=1.0 / resolution)
    freq_y = torch.fft.rfftfreq(resolution, d=1.0 / resolution)
    kx, ky = torch.meshgrid(freq_x, freq_y, indexing="ij")
    transfer = 1.0 / (1.0 + alpha_x * kx.square() + alpha_y * ky.square())
    field_hat = torch.fft.rfft2(field)
    return torch.fft.irfft2(field_hat * transfer, s=field.shape[-2:])


def directional_poisson_filter(
    field: torch.Tensor,
    alpha_parallel: float,
    alpha_transverse: float,
    theta: float,
) -> torch.Tensor:
    resolution = field.shape[-1]
    freq_x = torch.fft.fftfreq(resolution, d=1.0 / resolution)
    freq_y = torch.fft.rfftfreq(resolution, d=1.0 / resolution)
    kx, ky = torch.meshgrid(freq_x, freq_y, indexing="ij")
    k_parallel = math.cos(theta) * kx + math.sin(theta) * ky
    k_transverse = -math.sin(theta) * kx + math.cos(theta) * ky
    transfer = 1.0 / (
        1.0 + alpha_parallel * k_parallel.square() + alpha_transverse * k_transverse.square()
    )
    field_hat = torch.fft.rfft2(field)
    return torch.fft.irfft2(field_hat * transfer, s=field.shape[-2:])


def standardize_from_train(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    x_mean = x_train.mean(dim=(0, 1, 2), keepdim=True)
    x_std = x_train.std(dim=(0, 1, 2), keepdim=True).clamp_min(eps)
    y_mean = y_train.mean(dim=(0, 1, 2), keepdim=True)
    y_std = y_train.std(dim=(0, 1, 2), keepdim=True).clamp_min(eps)
    stats = {
        "x_mean": x_mean.squeeze().tolist(),
        "x_std": x_std.squeeze().tolist(),
        "y_mean": y_mean.squeeze().tolist(),
        "y_std": y_std.squeeze().tolist(),
    }
    return (
        (x_train - x_mean) / x_std,
        (y_train - y_mean) / y_std,
        (x_test - x_mean) / x_std,
        (y_test - y_mean) / y_std,
        stats,
    )


def save_dataset(
    dataset_dir: Path,
    spec: DatasetSpec,
    creation_script: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    train_samples: int = TRAIN_SAMPLES,
    test_samples: int = TEST_SAMPLES,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test, stats = standardize_from_train(
        x_train, y_train, x_test, y_test
    )
    payload = {
        "x_train": x_train.float(),
        "y_train": y_train.float(),
        "x_test": x_test.float(),
        "y_test": y_test.float(),
        "metadata": {
            "name": spec.name,
            "description": spec.description,
            "resolution": RESOLUTION,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "input_channels": int(x_train.shape[-1]),
            "output_channels": int(y_train.shape[-1]),
            "modes": spec.modes,
            "creation_script": creation_script,
            "normalization": stats,
        },
    }
    torch.save(payload, dataset_dir / "dataset.pt")
    with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["metadata"], handle, indent=2)
