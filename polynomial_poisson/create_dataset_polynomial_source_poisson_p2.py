from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

try:
    from polynomial_poisson.common import (
        GRID_X,
        GRID_Y,
        RESOLUTION,
        DatasetSpec,
        save_dataset,
    )
except ModuleNotFoundError:
    from common import (
        GRID_X,
        GRID_Y,
        RESOLUTION,
        DatasetSpec,
        save_dataset,
    )


DEFAULT_TRAIN_SAMPLES = 1000
DEFAULT_TEST_SAMPLES = 200
DEFAULT_K_MIN = 8.0
DEFAULT_K_MAX = 18.0
DEFAULT_NUM_MODES = 64
DEFAULT_SPECTRAL_DECAY = 0.75
EPS = 1e-6


def make_frequency_candidates(
    resolution: int,
    k_min: float,
    k_max: float,
) -> list[tuple[int, int]]:
    max_k = resolution // 2
    candidates: list[tuple[int, int]] = []
    for kx in range(-max_k + 1, max_k):
        for ky in range(-max_k + 1, max_k):
            if kx == 0 and ky == 0:
                continue
            radius = math.sqrt(kx**2 + ky**2)
            if k_min <= radius <= k_max:
                candidates.append((kx, ky))
    if not candidates:
        raise ValueError(
            f"No Fourier modes found for k_min={k_min}, k_max={k_max}, resolution={resolution}."
        )
    return candidates


def sample_random_fourier_field(
    generator: torch.Generator,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    candidates: list[tuple[int, int]],
    num_modes: int,
    spectral_decay: float,
) -> torch.Tensor:
    candidate_count = len(candidates)
    indices = torch.randint(
        low=0,
        high=candidate_count,
        size=(num_modes,),
        generator=generator,
    )
    field = torch.zeros((grid_x.shape[0], grid_x.shape[1]), dtype=torch.float32)
    for idx in indices.tolist():
        kx, ky = candidates[idx]
        radius = math.sqrt(kx**2 + ky**2)
        phase = 2.0 * math.pi * torch.rand((), generator=generator).item()
        coeff = torch.randn((), generator=generator).item()
        amplitude = coeff / (radius**spectral_decay)
        angle = 2.0 * math.pi * (kx * grid_x + ky * grid_y) + phase
        field = field + amplitude * torch.cos(angle)
    field = field - field.mean()
    field = field / (field.std() + EPS)
    return field


def solve_periodic_poisson(source: torch.Tensor) -> torch.Tensor:
    resolution = source.shape[0]
    source = source - source.mean()
    source_hat = torch.fft.fft2(source)
    freq = torch.fft.fftfreq(resolution, d=1.0 / resolution)
    kx, ky = torch.meshgrid(freq, freq, indexing="ij")
    laplace_eigs = 4.0 * math.pi**2 * (kx.square() + ky.square())
    v_hat = torch.zeros_like(source_hat)
    nonzero = laplace_eigs > 0
    v_hat[nonzero] = source_hat[nonzero] / laplace_eigs[nonzero]
    return torch.fft.ifft2(v_hat).real


def build_split(
    n_samples: int,
    seed: int,
    degree: int,
    resolution: int,
    k_min: float,
    k_max: float,
    num_modes: int,
    spectral_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    grid_x, grid_y = GRID_X[:resolution, :resolution], GRID_Y[:resolution, :resolution]
    candidates = make_frequency_candidates(resolution, k_min, k_max)

    x = torch.zeros((n_samples, resolution, resolution, degree), dtype=torch.float32)
    y = torch.zeros((n_samples, resolution, resolution, 1), dtype=torch.float32)

    for sample_idx in range(n_samples):
        fields = []
        for channel_idx in range(degree):
            field = sample_random_fourier_field(
                generator=generator,
                grid_x=grid_x,
                grid_y=grid_y,
                candidates=candidates,
                num_modes=num_modes,
                spectral_decay=spectral_decay,
            )
            fields.append(field)
            x[sample_idx, ..., channel_idx] = field

        source = torch.ones((resolution, resolution), dtype=torch.float32)
        for field in fields:
            source = source * field
        source = source - source.mean()
        y[sample_idx, ..., 0] = solve_periodic_poisson(source)

    return x, y


def normalize_targets(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    mean = y_train.mean()
    std = y_train.std()
    return (
        (y_train - mean) / (std + EPS),
        (y_test - mean) / (std + EPS),
        {
            "target_mean": float(mean.item()),
            "target_std": float(std.item()),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="generated")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--k-min", type=float, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max", type=float, default=DEFAULT_K_MAX)
    parser.add_argument("--num-modes", type=int, default=DEFAULT_NUM_MODES)
    parser.add_argument("--spectral-decay", type=float, default=DEFAULT_SPECTRAL_DECAY)
    parser.add_argument("--train-seed", type=int, default=2101)
    parser.add_argument("--test-seed", type=int, default=2143)
    parser.add_argument("--no-target-normalization", action="store_true")
    args = parser.parse_args()

    if args.degree < 1:
        raise ValueError("degree should be at least 1.")

    dataset_name = f"polynomial_source_poisson_p{args.degree}"
    output_dir = Path(args.output_root) / dataset_name

    x_train, y_train = build_split(
        n_samples=args.train_samples,
        seed=args.train_seed,
        degree=args.degree,
        resolution=args.resolution,
        k_min=args.k_min,
        k_max=args.k_max,
        num_modes=args.num_modes,
        spectral_decay=args.spectral_decay,
    )
    x_test, y_test = build_split(
        n_samples=args.test_samples,
        seed=args.test_seed,
        degree=args.degree,
        resolution=args.resolution,
        k_min=args.k_min,
        k_max=args.k_max,
        num_modes=args.num_modes,
        spectral_decay=args.spectral_decay,
    )

    normalization_stats = {}
    if not args.no_target_normalization:
        y_train, y_test, normalization_stats = normalize_targets(y_train, y_test)

    metadata = {
        "name": dataset_name,
        "description": (
            "Polynomial-Source Poisson dataset. Inputs are p independent band-limited random Fourier fields. "
            + (
                "For p=1, the target solves the periodic Poisson equation -Delta v = u_1 - mean(u_1)."
                if args.degree == 1
                else "The target solves the periodic Poisson equation -Delta v = prod_j u_j - mean(prod_j u_j)."
            )
        ),
        "degree": args.degree,
        "resolution": args.resolution,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "input_shape": list(x_train.shape[1:]),
        "target_shape": list(y_train.shape[1:]),
        "k_min": args.k_min,
        "k_max": args.k_max,
        "num_modes": args.num_modes,
        "spectral_decay": args.spectral_decay,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "target_normalized": not args.no_target_normalization,
        **normalization_stats,
    }

    spec = DatasetSpec(
        name=dataset_name,
        description=metadata["description"],
        modes=16,
    )
    save_dataset(
        output_dir,
        spec,
        Path(__file__).name,
        x_train,
        y_train,
        x_test,
        y_test,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved dataset to: {output_dir}")


if __name__ == "__main__":
    main()
