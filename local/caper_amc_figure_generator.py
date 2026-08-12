#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPER-AMC paper figure generator
================================

This script is designed for the model-training file:
    1111caper_amc_local.py

It loads the saved EMA checkpoint and recreates the exact test split used by
that training script. It then generates publication-ready PNG and PDF figures,
as well as CSV/JSON tables for paper writing.

Typical usage (run from the folder containing the training script, dataset,
and caper_amc_results directory):

    python caper_amc_figure_generator.py

Explicit paths:

    python caper_amc_figure_generator.py \
        --training-script 1111caper_amc_local.py \
        --checkpoint caper_amc_results/caper_amc_best_vram.pth \
        --dataset RML2016.10a_dict.pkl \
        --output-dir caper_amc_results/paper_figures

Optional training curves:

    python caper_amc_figure_generator.py \
        --history caper_amc_results/training_history.csv

Expected history columns (any available subset is accepted):
    epoch, train_loss, train_accuracy, val_accuracy, val_macro_f1,
    router_temperature, epoch_time

Important limitation:
The supplied training code does not save epoch history. Therefore, all
post-training figures can be generated from the checkpoint, but training
curves require a separately saved history CSV.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
import time
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROUTE_NAMES = ["full", "causal", "local"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from a trained CAPER-AMC checkpoint."
    )
    parser.add_argument(
        "--training-script",
        type=Path,
        default=Path("1111caper_amc_local.py"),
        help="Path to the CAPER-AMC training Python file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("caper_amc_results/caper_amc_best_vram.pth"),
        help="Path to the saved EMA checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to a RadioML .pkl/.pickle file or ZIP containing one. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("caper_amc_results/paper_figures"),
        help="Directory for generated figures and tables.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Optional training_history.csv for training-curve figures.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Evaluation device.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--latency-batch-size",
        type=int,
        default=512,
        help="Batch size for latency measurement.",
    )
    parser.add_argument(
        "--latency-repeats",
        type=int,
        default=40,
        help="Timed repetitions for each routing mode.",
    )
    parser.add_argument(
        "--skip-route-benchmark",
        action="store_true",
        help="Skip forced full/causal/local accuracy and latency evaluation.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Save PNG only instead of both PNG and PDF.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_training_module(path: Path):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Training script not found: {path}")

    spec = importlib.util.spec_from_file_location("caper_training_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import training script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = ["CAPERAMC", "MIN_SNR", "MAX_SNR", "TRAIN_RATIO", "VAL_RATIO", "SEED"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Training script is missing required names: {missing}")
    return module


def decode_modulation_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def resolve_dataset_path(
    requested: Optional[Path],
    search_roots: Sequence[Path],
    extraction_dir: Path,
) -> Path:
    candidates: list[Path] = []

    if requested is not None:
        candidates = [requested.expanduser().resolve()]
    else:
        for root in search_roots:
            root = root.expanduser().resolve()
            if not root.exists():
                continue
            candidates.extend(sorted(root.glob("*.pkl")))
            candidates.extend(sorted(root.glob("*.pickle")))
            candidates.extend(sorted(root.glob("*.zip")))

    if not candidates:
        raise FileNotFoundError(
            "No dataset was found. Supply --dataset with a .pkl, .pickle, or .zip file."
        )

    dataset = candidates[0]
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    if dataset.suffix.lower() != ".zip":
        return dataset

    extraction_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset, "r") as archive:
        archive.extractall(extraction_dir)

    extracted = sorted(extraction_dir.rglob("*.pkl")) + sorted(
        extraction_dir.rglob("*.pickle")
    )
    if not extracted:
        raise FileNotFoundError(f"No .pkl/.pickle dataset found inside ZIP: {dataset}")
    return extracted[0]


def load_test_split_exactly(
    dataset_path: Path,
    min_snr: int,
    max_snr: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Recreate only the test split using the same RNG/order as the training file."""
    print(f"Loading exact test split from: {dataset_path}")
    with dataset_path.open("rb") as file:
        raw_data = pickle.load(file, encoding="latin1")

    class_names = sorted(
        {decode_modulation_name(modulation) for modulation, _ in raw_data.keys()}
    )
    class_to_index = {name: index for index, name in enumerate(class_names)}
    generator = np.random.default_rng(seed)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []

    for (raw_modulation, raw_snr), samples in raw_data.items():
        snr_value = int(raw_snr)
        if not (min_snr <= snr_value <= max_snr):
            continue

        samples = np.asarray(samples, dtype=np.float32)
        indices = generator.permutation(len(samples))
        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)
        test_indices = indices[val_end:]

        if len(test_indices) == 0:
            continue

        class_index = class_to_index[decode_modulation_name(raw_modulation)]
        x_parts.append(samples[test_indices])
        y_parts.append(np.full(len(test_indices), class_index, dtype=np.int64))
        snr_parts.append(np.full(len(test_indices), snr_value, dtype=np.float32))

    del raw_data

    if not x_parts:
        raise RuntimeError("The selected SNR range produced an empty test split.")

    x_np = np.concatenate(x_parts, axis=0)
    y_np = np.concatenate(y_parts, axis=0)
    snr_np = np.concatenate(snr_parts, axis=0)

    print(
        f"Test samples: {len(y_np):,} | Classes: {len(class_names)} | "
        f"SNR range: {int(snr_np.min())} to {int(snr_np.max())} dB"
    )

    return {
        "X": torch.from_numpy(x_np).to(device=device, dtype=torch.float32),
        "y": torch.from_numpy(y_np).to(device=device, dtype=torch.long),
        "snr": torch.from_numpy(snr_np).to(device=device, dtype=torch.float32),
        "class_names": class_names,
    }


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(training_module, checkpoint: Dict[str, Any], device: torch.device):
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise KeyError("Checkpoint does not contain 'class_names'.")

    model = training_module.CAPERAMC(
        class_names=class_names,
        min_snr=training_module.MIN_SNR,
        max_snr=training_module.MAX_SNR,
        input_length=128,
        d_model=getattr(training_module, "D_MODEL", 128),
        n_heads=getattr(training_module, "N_HEADS", 4),
        route_layers=getattr(training_module, "ATTENTION_LAYERS_PER_ROUTE", 2),
        d_ff=getattr(training_module, "D_FF", 384),
        dropout=getattr(training_module, "DROPOUT", 0.15),
        local_window=getattr(training_module, "LOCAL_WINDOW", 9),
        routing_mode=getattr(training_module, "ROUTING_MODE", "conditional_top1"),
    ).to(device)

    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint has neither 'ema_state_dict' nor 'state_dict'.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def forced_router(model, route_name: Optional[str]):
    """Force one route by replacing router logits through a forward hook."""
    if route_name is None:
        yield
        return
    if route_name not in ROUTE_NAMES:
        raise ValueError(f"Unknown route: {route_name}")

    route_index = ROUTE_NAMES.index(route_name)

    def hook(_module, _inputs, output):
        forced = torch.full_like(output, -1.0e4)
        forced[:, route_index] = 1.0e4
        return forced

    handle = model.router.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def collect_predictions(
    model,
    data: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    forced_route_name: Optional[str] = None,
    collect_details: bool = True,
) -> Dict[str, np.ndarray | float]:
    model.eval()
    x_all, y_all, snr_all = data["X"], data["y"], data["snr"]

    target_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    snr_parts: list[np.ndarray] = []
    pred_snr_parts: list[np.ndarray] = []
    route_prob_parts: list[np.ndarray] = []
    route_parts: list[np.ndarray] = []
    uncertainty_parts: list[np.ndarray] = []

    synchronize(device)
    start = time.perf_counter()

    with forced_router(model, forced_route_name):
        for start_index in range(0, len(y_all), batch_size):
            end_index = start_index + batch_size
            inputs = x_all[start_index:end_index]
            targets = y_all[start_index:end_index]
            true_snr = snr_all[start_index:end_index]

            # This is the same deterministic evaluation normalization used by training code.
            inputs = inputs - inputs.mean(dim=-1, keepdim=True)
            rms = torch.sqrt(torch.mean(inputs.square(), dim=-1, keepdim=True)).clamp_min(1e-6)
            inputs = inputs / rms

            with amp_context(device):
                outputs = model(inputs, use_gumbel=False)

            logits = outputs["logits"].float()
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            target_parts.append(targets.detach().cpu().numpy())
            pred_parts.append(predictions.detach().cpu().numpy())
            snr_parts.append(true_snr.detach().cpu().numpy())

            if collect_details:
                coarse_probabilities = torch.softmax(outputs["coarse_logits"].float(), dim=1)
                uncertainty = -torch.sum(
                    coarse_probabilities
                    * torch.log(coarse_probabilities.clamp_min(1e-8)),
                    dim=1,
                ) / math.log(model.num_classes)

                predicted_snr = model.normalized_snr_to_db(
                    outputs["snr_normalized"].float()
                ).squeeze(1)
                route_probabilities = outputs["routing_probabilities"].float()
                selected_routes = outputs["route_assignments"].argmax(dim=1)

                prob_parts.append(probabilities.detach().cpu().numpy())
                pred_snr_parts.append(predicted_snr.detach().cpu().numpy())
                route_prob_parts.append(route_probabilities.detach().cpu().numpy())
                route_parts.append(selected_routes.detach().cpu().numpy())
                uncertainty_parts.append(uncertainty.detach().cpu().numpy())

    synchronize(device)
    elapsed = time.perf_counter() - start

    result: Dict[str, np.ndarray | float] = {
        "targets": np.concatenate(target_parts),
        "predictions": np.concatenate(pred_parts),
        "true_snr": np.concatenate(snr_parts),
        "elapsed_seconds": elapsed,
    }
    if collect_details:
        result.update(
            {
                "probabilities": np.concatenate(prob_parts),
                "predicted_snr": np.concatenate(pred_snr_parts),
                "route_probabilities": np.concatenate(route_prob_parts),
                "selected_routes": np.concatenate(route_parts),
                "uncertainty": np.concatenate(uncertainty_parts),
            }
        )
    return result


@torch.inference_mode()
def benchmark_latency(
    model,
    x: torch.Tensor,
    device: torch.device,
    route_name: Optional[str],
    repeats: int,
) -> float:
    """Return median forward latency in milliseconds per sample."""
    x = x - x.mean(dim=-1, keepdim=True)
    rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True)).clamp_min(1e-6)
    x = x / rms

    warmups = min(10, max(3, repeats // 4))
    timings: list[float] = []

    with forced_router(model, route_name):
        for _ in range(warmups):
            with amp_context(device):
                _ = model(x, use_gumbel=False)
        synchronize(device)

        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            with amp_context(device):
                _ = model(x, use_gumbel=False)
            synchronize(device)
            timings.append(time.perf_counter() - start)

    return 1000.0 * float(np.median(timings)) / x.shape[0]


def calculate_ece(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, pd.DataFrame]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    ece = 0.0
    for index in range(n_bins):
        left, right = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        count = int(mask.sum())
        if count == 0:
            mean_confidence = np.nan
            bin_accuracy = np.nan
        else:
            mean_confidence = float(confidences[mask].mean())
            bin_accuracy = float(correct[mask].mean())
            ece += (count / len(targets)) * abs(bin_accuracy - mean_confidence)

        rows.append(
            {
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": bin_accuracy,
            }
        )

    return float(ece), pd.DataFrame(rows)


def make_analysis_tables(
    results: Dict[str, np.ndarray | float],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    targets = np.asarray(results["targets"])
    predictions = np.asarray(results["predictions"])
    probabilities = np.asarray(results["probabilities"])
    true_snr = np.asarray(results["true_snr"])
    predicted_snr = np.asarray(results["predicted_snr"])
    route_probabilities = np.asarray(results["route_probabilities"])
    selected_routes = np.asarray(results["selected_routes"])
    uncertainty = np.asarray(results["uncertainty"])

    labels = np.arange(len(class_names))
    report_dict = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    per_class_df = report_df.loc[list(class_names), ["precision", "recall", "f1-score", "support"]].copy()
    per_class_df.index.name = "modulation"
    per_class_df.reset_index(inplace=True)

    cm_counts = confusion_matrix(targets, predictions, labels=labels)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm_counts,
        row_sums,
        out=np.zeros_like(cm_counts, dtype=float),
        where=row_sums != 0,
    )

    snr_rows = []
    route_snr_rows = []
    uncertainty_rows = []
    snr_estimation_rows = []

    for snr_value in sorted(np.unique(true_snr)):
        mask = true_snr == snr_value
        snr_targets = targets[mask]
        snr_predictions = predictions[mask]
        snr_rows.append(
            {
                "snr_db": int(snr_value),
                "samples": int(mask.sum()),
                "accuracy_percent": 100.0 * accuracy_score(snr_targets, snr_predictions),
                "macro_f1": f1_score(
                    snr_targets,
                    snr_predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

        route_row = {"snr_db": int(snr_value), "samples": int(mask.sum())}
        for route_index, route_name in enumerate(ROUTE_NAMES):
            route_row[f"selected_{route_name}"] = float(
                np.mean(selected_routes[mask] == route_index)
            )
            route_row[f"probability_{route_name}"] = float(
                route_probabilities[mask, route_index].mean()
            )
        route_snr_rows.append(route_row)

        uncertainty_rows.append(
            {
                "snr_db": int(snr_value),
                "mean_uncertainty": float(uncertainty[mask].mean()),
                "std_uncertainty": float(uncertainty[mask].std()),
                "mean_confidence": float(probabilities[mask].max(axis=1).mean()),
            }
        )

        snr_error = np.abs(predicted_snr[mask] - true_snr[mask])
        snr_estimation_rows.append(
            {
                "snr_db": int(snr_value),
                "samples": int(mask.sum()),
                "predicted_snr_mean": float(predicted_snr[mask].mean()),
                "predicted_snr_std": float(predicted_snr[mask].std()),
                "mae_db": float(snr_error.mean()),
                "rmse_db": float(np.sqrt(np.mean((predicted_snr[mask] - true_snr[mask]) ** 2))),
            }
        )

    routing_class_rows = []
    for class_index, class_name in enumerate(class_names):
        mask = targets == class_index
        row = {"modulation": class_name, "samples": int(mask.sum())}
        for route_index, route_name in enumerate(ROUTE_NAMES):
            row[f"selected_{route_name}"] = float(
                np.mean(selected_routes[mask] == route_index)
            )
            row[f"probability_{route_name}"] = float(
                route_probabilities[mask, route_index].mean()
            )
        routing_class_rows.append(row)

    ece, calibration_df = calculate_ece(probabilities, targets)
    confidence = probabilities.max(axis=1)

    predictions_df = pd.DataFrame(
        {
            "true_index": targets,
            "true_modulation": [class_names[index] for index in targets],
            "predicted_index": predictions,
            "predicted_modulation": [class_names[index] for index in predictions],
            "correct": targets == predictions,
            "confidence": confidence,
            "true_snr_db": true_snr,
            "predicted_snr_db": predicted_snr,
            "uncertainty": uncertainty,
            "selected_route": [ROUTE_NAMES[index] for index in selected_routes],
            "route_prob_full": route_probabilities[:, 0],
            "route_prob_causal": route_probabilities[:, 1],
            "route_prob_local": route_probabilities[:, 2],
        }
    )

    summary = {
        "test_samples": int(len(targets)),
        "accuracy_percent": 100.0 * accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(targets, predictions, average="weighted", zero_division=0),
        "ece_15_bins": ece,
        "snr_mae_db": float(np.mean(np.abs(predicted_snr - true_snr))),
        "snr_rmse_db": float(np.sqrt(np.mean((predicted_snr - true_snr) ** 2))),
        "overall_route_usage": {
            route_name: float(np.mean(selected_routes == route_index))
            for route_index, route_name in enumerate(ROUTE_NAMES)
        },
    }

    return {
        "summary": summary,
        "predictions": predictions_df,
        "per_class": per_class_df,
        "classification_report": report_df,
        "confusion_counts": cm_counts,
        "confusion_normalized": cm_normalized,
        "snr_metrics": pd.DataFrame(snr_rows),
        "routing_by_snr": pd.DataFrame(route_snr_rows),
        "routing_by_class": pd.DataFrame(routing_class_rows),
        "uncertainty_by_snr": pd.DataFrame(uncertainty_rows),
        "snr_estimation": pd.DataFrame(snr_estimation_rows),
        "calibration": calibration_df,
    }


def save_figure(fig: plt.Figure, output_stem: Path, save_pdf: bool) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, x, y, w, h, text, fontsize=9, linewidth=1.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=linewidth,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def draw_arrow(ax, start: Tuple[float, float], end: Tuple[float, float]):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="black",
    )
    ax.add_patch(arrow)


def figure_architecture(output_dir: Path, class_count: int, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, 0.04, 0.79, 0.16, 0.11, "I/Q input\n2 × 128")
    draw_box(ax, 0.25, 0.79, 0.18, 0.11, "Physics features\nI, Q, |x|, Δ|x|,\nphase sin/cos")
    draw_box(ax, 0.48, 0.79, 0.20, 0.11, "Multi-scale CNN\nk = 3, 5, 7\n+ squeeze-excite")
    draw_box(ax, 0.74, 0.79, 0.20, 0.11, "Token sequence\n64 × 128\n+ position embedding")

    draw_arrow(ax, (0.20, 0.845), (0.25, 0.845))
    draw_arrow(ax, (0.43, 0.845), (0.48, 0.845))
    draw_arrow(ax, (0.68, 0.845), (0.74, 0.845))

    draw_box(ax, 0.04, 0.55, 0.20, 0.11, "Spectral context\nFFT power → MLP")
    draw_box(ax, 0.31, 0.55, 0.20, 0.11, "Preliminary fusion\ncoarse logits + SNR\n+ uncertainty")
    draw_box(ax, 0.58, 0.55, 0.18, 0.11, "Conditional router\nTop-1 route selection")

    draw_arrow(ax, (0.12, 0.79), (0.12, 0.66))
    draw_arrow(ax, (0.24, 0.605), (0.31, 0.605))
    draw_arrow(ax, (0.84, 0.79), (0.84, 0.71))
    draw_arrow(ax, (0.84, 0.71), (0.41, 0.66))
    draw_arrow(ax, (0.51, 0.605), (0.58, 0.605))

    route_y = 0.31
    draw_box(ax, 0.18, route_y, 0.18, 0.11, "Full attention\n2 Transformer blocks")
    draw_box(ax, 0.41, route_y, 0.18, 0.11, "Causal attention\n2 Transformer blocks")
    draw_box(ax, 0.64, route_y, 0.18, 0.11, "Local attention\nwindow = 9")

    draw_arrow(ax, (0.67, 0.55), (0.27, route_y + 0.11))
    draw_arrow(ax, (0.67, 0.55), (0.50, route_y + 0.11))
    draw_arrow(ax, (0.67, 0.55), (0.73, route_y + 0.11))

    draw_box(ax, 0.15, 0.08, 0.22, 0.11, "Dual temporal pooling\nattention pooling + mean")
    draw_box(ax, 0.43, 0.08, 0.20, 0.11, "Representation fusion\nwith spectral context")
    draw_box(ax, 0.69, 0.08, 0.25, 0.11, f"Classifier + pair experts\nQAM16/QAM64 and AM-DSB/WBFM\n→ {class_count} classes")

    for x in (0.27, 0.50, 0.73):
        draw_arrow(ax, (x, route_y), (0.26, 0.19))
    draw_arrow(ax, (0.37, 0.135), (0.43, 0.135))
    draw_arrow(ax, (0.63, 0.135), (0.69, 0.135))

    ax.set_title("CAPER-AMC Architecture Implemented by the Training Code", fontsize=15, pad=12)
    ax.text(
        0.5,
        0.015,
        "Note: the embedding head and supervised-contrastive flag are defined in the training file, "
        "but the current loss does not use them.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    save_figure(fig, output_dir / "fig01_caper_amc_architecture", save_pdf)


def figure_input_signal(
    data: Dict[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
    save_pdf: bool,
) -> None:
    snr = data["snr"].detach().cpu().numpy()
    y = data["y"].detach().cpu().numpy()
    max_snr = snr.max()
    chosen_index = int(np.where(snr == max_snr)[0][0])
    signal = data["X"][chosen_index].detach().cpu().numpy()
    class_name = class_names[int(y[chosen_index])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    time_axis = np.arange(signal.shape[-1])
    axes[0].plot(time_axis, signal[0], label="I")
    axes[0].plot(time_axis, signal[1], label="Q")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("I/Q Waveforms")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].scatter(signal[0], signal[1], s=12, alpha=0.7)
    axes[1].axhline(0, linewidth=0.8)
    axes[1].axvline(0, linewidth=0.8)
    axes[1].set_xlabel("In-phase (I)")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].set_title("I/Q Constellation Trace")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"Example test signal: {class_name}, SNR = {int(max_snr)} dB")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig02_example_iq_signal", save_pdf)


def figure_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_dir: Path,
    save_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Normalized frequency")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted modulation")
    ax.set_ylabel("True modulation")
    ax.set_title("Normalized Confusion Matrix")

    threshold = 0.5
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    save_figure(fig, output_dir / "fig03_normalized_confusion_matrix", save_pdf)


def figure_per_class_metrics(
    per_class: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    x = np.arange(len(per_class))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width, per_class["precision"], width, label="Precision")
    ax.bar(x, per_class["recall"], width, label="Recall")
    ax.bar(x + width, per_class["f1-score"], width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(per_class["modulation"], rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Modulation class")
    ax.set_title("Per-Modulation Classification Performance")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir / "fig04_per_modulation_metrics", save_pdf)


def figure_performance_vs_snr(
    snr_metrics: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        snr_metrics["snr_db"],
        snr_metrics["accuracy_percent"],
        marker="o",
        label="Accuracy",
    )
    ax.plot(
        snr_metrics["snr_db"],
        100.0 * snr_metrics["macro_f1"],
        marker="s",
        label="Macro-F1",
    )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(0, 102)
    ax.set_xticks(snr_metrics["snr_db"])
    ax.set_title("CAPER-AMC Performance versus SNR")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "fig05_performance_vs_snr", save_pdf)


def figure_routing_vs_snr(
    routing: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    x = np.arange(len(routing))
    bottom = np.zeros(len(routing))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for route_name in ROUTE_NAMES:
        values = routing[f"selected_{route_name}"].to_numpy()
        ax.bar(x, values, bottom=bottom, label=route_name.capitalize())
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(routing["snr_db"].astype(int))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Fraction of test samples")
    ax.set_ylim(0, 1.02)
    ax.set_title("Conditional Attention Route Selection versus SNR")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir / "fig06_routing_vs_snr", save_pdf)


def figure_routing_by_class(
    routing: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    matrix = routing[[f"selected_{name}" for name in ROUTE_NAMES]].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, max(5, 0.45 * len(routing))))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Selection fraction")
    ax.set_xticks(np.arange(len(ROUTE_NAMES)))
    ax.set_xticklabels([name.capitalize() for name in ROUTE_NAMES])
    ax.set_yticks(np.arange(len(routing)))
    ax.set_yticklabels(routing["modulation"])
    ax.set_xlabel("Attention route")
    ax.set_ylabel("True modulation")
    ax.set_title("Route Selection by Modulation Class")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.5 else "black",
            )
    fig.tight_layout()
    save_figure(fig, output_dir / "fig07_routing_by_modulation", save_pdf)


def figure_calibration(
    calibration: pd.DataFrame,
    ece: float,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    valid = calibration[calibration["count"] > 0]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(
        valid["mean_confidence"],
        valid["accuracy"],
        marker="o",
        label=f"CAPER-AMC (ECE={ece:.4f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability Diagram")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "fig08_reliability_diagram", save_pdf)


def figure_snr_estimation(
    results: Dict[str, np.ndarray | float],
    snr_estimation: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    true_snr = np.asarray(results["true_snr"])
    predicted_snr = np.asarray(results["predicted_snr"])
    rng = np.random.default_rng(42)
    sample_count = min(5000, len(true_snr))
    indices = rng.choice(len(true_snr), size=sample_count, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(true_snr[indices], predicted_snr[indices], s=8, alpha=0.25)
    low = min(true_snr.min(), predicted_snr.min())
    high = max(true_snr.max(), predicted_snr.max())
    axes[0].plot([low, high], [low, high], linestyle="--", label="Ideal")
    axes[0].set_xlabel("True SNR (dB)")
    axes[0].set_ylabel("Predicted SNR (dB)")
    axes[0].set_title("Predicted versus True SNR")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        snr_estimation["snr_db"],
        snr_estimation["mae_db"],
        marker="o",
    )
    axes[1].set_xticks(snr_estimation["snr_db"])
    axes[1].set_xlabel("True SNR (dB)")
    axes[1].set_ylabel("Mean absolute error (dB)")
    axes[1].set_title("SNR Estimation Error")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    save_figure(fig, output_dir / "fig09_snr_estimation", save_pdf)


def figure_uncertainty_vs_snr(
    uncertainty: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(
        uncertainty["snr_db"],
        uncertainty["mean_uncertainty"],
        marker="o",
        label="Normalized entropy uncertainty",
    )
    ax.plot(
        uncertainty["snr_db"],
        1.0 - uncertainty["mean_confidence"],
        marker="s",
        label="1 − mean confidence",
    )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Uncertainty")
    ax.set_xticks(uncertainty["snr_db"])
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Uncertainty versus SNR")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "fig10_uncertainty_vs_snr", save_pdf)


def figure_route_tradeoff(
    benchmark: pd.DataFrame,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        benchmark["latency_ms_per_sample"],
        benchmark["accuracy_percent"],
        s=80,
    )
    for _, row in benchmark.iterrows():
        ax.annotate(
            row["mode"],
            (row["latency_ms_per_sample"], row["accuracy_percent"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Median inference latency (ms/sample)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy–Latency Trade-off by Attention Route")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir / "fig11_accuracy_latency_tradeoff", save_pdf)


def figure_training_history(
    history_path: Path,
    output_dir: Path,
    save_pdf: bool,
) -> None:
    history = pd.read_csv(history_path)
    if "epoch" not in history.columns:
        history.insert(0, "epoch", np.arange(1, len(history) + 1))

    available = set(history.columns)
    has_accuracy = bool({"train_accuracy", "val_accuracy"} & available)
    has_loss = "train_loss" in available
    has_f1 = "val_macro_f1" in available

    if not (has_accuracy or has_loss or has_f1):
        print(
            f"Skipping training curves: {history_path} has no recognized metric columns."
        )
        return

    figure_count = int(has_accuracy) + int(has_loss or has_f1)
    fig, axes = plt.subplots(1, figure_count, figsize=(6 * figure_count, 4.8))
    if figure_count == 1:
        axes = [axes]
    axis_index = 0

    if has_accuracy:
        ax = axes[axis_index]
        if "train_accuracy" in history:
            ax.plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
        if "val_accuracy" in history:
            ax.plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Training and Validation Accuracy")
        ax.grid(alpha=0.25)
        ax.legend()
        axis_index += 1

    if has_loss or has_f1:
        ax = axes[axis_index]
        if has_loss:
            ax.plot(history["epoch"], history["train_loss"], label="Train loss")
        if has_f1:
            ax.plot(history["epoch"], history["val_macro_f1"], label="Validation Macro-F1")
        ax.set_xlabel("Epoch")
        ax.set_title("Training Progress")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    save_figure(fig, output_dir / "fig12_training_curves", save_pdf)


def save_tables(
    tables: Dict[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables["predictions"].to_csv(output_dir / "test_predictions.csv", index=False)
    tables["per_class"].to_csv(output_dir / "per_class_metrics.csv", index=False)
    tables["classification_report"].to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(
        tables["confusion_counts"], index=class_names, columns=class_names
    ).to_csv(output_dir / "confusion_matrix_counts.csv")
    pd.DataFrame(
        tables["confusion_normalized"], index=class_names, columns=class_names
    ).to_csv(output_dir / "confusion_matrix_normalized.csv")
    tables["snr_metrics"].to_csv(output_dir / "performance_by_snr.csv", index=False)
    tables["routing_by_snr"].to_csv(output_dir / "routing_by_snr.csv", index=False)
    tables["routing_by_class"].to_csv(output_dir / "routing_by_modulation.csv", index=False)
    tables["uncertainty_by_snr"].to_csv(output_dir / "uncertainty_by_snr.csv", index=False)
    tables["snr_estimation"].to_csv(output_dir / "snr_estimation_by_snr.csv", index=False)
    tables["calibration"].to_csv(output_dir / "calibration_bins.csv", index=False)
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(tables["summary"], file, indent=2)


def run_route_benchmark(
    model,
    data: Dict[str, Any],
    conditional_results: Dict[str, np.ndarray | float],
    device: torch.device,
    eval_batch_size: int,
    latency_batch_size: int,
    latency_repeats: int,
) -> pd.DataFrame:
    sample_batch = data["X"][: min(latency_batch_size, len(data["y"]))]
    rows = []

    modes: list[Tuple[str, Optional[str]]] = [
        ("Conditional Top-1", None),
        ("Forced Full", "full"),
        ("Forced Causal", "causal"),
        ("Forced Local", "local"),
    ]

    for display_name, route_name in modes:
        print(f"Evaluating route mode: {display_name}")
        if route_name is None:
            mode_results = conditional_results
        else:
            mode_results = collect_predictions(
                model,
                data,
                device,
                batch_size=eval_batch_size,
                forced_route_name=route_name,
                collect_details=False,
            )

        targets = np.asarray(mode_results["targets"])
        predictions = np.asarray(mode_results["predictions"])
        latency = benchmark_latency(
            model,
            sample_batch,
            device,
            route_name=route_name,
            repeats=latency_repeats,
        )
        rows.append(
            {
                "mode": display_name,
                "forced_route": route_name or "conditional",
                "accuracy_percent": 100.0 * accuracy_score(targets, predictions),
                "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
                "latency_ms_per_sample": latency,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pdf = not args.no_pdf

    print(f"Using device: {device}")
    training_script = args.training_script.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    training_module = import_training_module(training_script)

    dataset_path = resolve_dataset_path(
        args.dataset,
        search_roots=[Path.cwd(), training_script.parent, checkpoint_path.parent.parent],
        extraction_dir=output_dir / "extracted_dataset",
    )

    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(training_module, checkpoint, device)
    checkpoint_class_names = list(checkpoint["class_names"])

    data = load_test_split_exactly(
        dataset_path=dataset_path,
        min_snr=int(training_module.MIN_SNR),
        max_snr=int(training_module.MAX_SNR),
        train_ratio=float(training_module.TRAIN_RATIO),
        val_ratio=float(training_module.VAL_RATIO),
        seed=int(training_module.SEED),
        device=device,
    )

    if list(data["class_names"]) != checkpoint_class_names:
        raise ValueError(
            "Dataset class names do not match checkpoint class names.\n"
            f"Dataset: {data['class_names']}\nCheckpoint: {checkpoint_class_names}"
        )

    print("Running conditional Top-1 test evaluation...")
    results = collect_predictions(
        model,
        data,
        device,
        batch_size=args.batch_size,
        forced_route_name=None,
        collect_details=True,
    )
    tables = make_analysis_tables(results, checkpoint_class_names)
    save_tables(tables, checkpoint_class_names, output_dir)

    figure_architecture(output_dir, len(checkpoint_class_names), save_pdf)
    figure_input_signal(data, checkpoint_class_names, output_dir, save_pdf)
    figure_confusion_matrix(
        tables["confusion_normalized"], checkpoint_class_names, output_dir, save_pdf
    )
    figure_per_class_metrics(tables["per_class"], output_dir, save_pdf)
    figure_performance_vs_snr(tables["snr_metrics"], output_dir, save_pdf)
    figure_routing_vs_snr(tables["routing_by_snr"], output_dir, save_pdf)
    figure_routing_by_class(tables["routing_by_class"], output_dir, save_pdf)
    figure_calibration(
        tables["calibration"], tables["summary"]["ece_15_bins"], output_dir, save_pdf
    )
    figure_snr_estimation(results, tables["snr_estimation"], output_dir, save_pdf)
    figure_uncertainty_vs_snr(tables["uncertainty_by_snr"], output_dir, save_pdf)

    if not args.skip_route_benchmark:
        benchmark = run_route_benchmark(
            model=model,
            data=data,
            conditional_results=results,
            device=device,
            eval_batch_size=args.batch_size,
            latency_batch_size=args.latency_batch_size,
            latency_repeats=args.latency_repeats,
        )
        benchmark.to_csv(output_dir / "route_accuracy_latency_benchmark.csv", index=False)
        figure_route_tradeoff(benchmark, output_dir, save_pdf)

    history_path = args.history
    if history_path is None:
        default_history = checkpoint_path.parent / "training_history.csv"
        if default_history.exists():
            history_path = default_history
    if history_path is not None and history_path.expanduser().exists():
        figure_training_history(history_path.expanduser().resolve(), output_dir, save_pdf)
    else:
        print(
            "Training curves were not generated because training_history.csv was not found. "
            "The supplied training code prints epoch metrics but does not save them."
        )

    summary = tables["summary"]
    print("\nGenerated paper figures and tables successfully.")
    print(f"Output directory: {output_dir}")
    print(f"Test accuracy: {summary['accuracy_percent']:.2f}%")
    print(f"Macro-F1: {summary['macro_f1']:.4f}")
    print(f"ECE: {summary['ece_15_bins']:.4f}")
    print(f"SNR MAE: {summary['snr_mae_db']:.3f} dB")


if __name__ == "__main__":
    main()
