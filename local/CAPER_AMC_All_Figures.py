# -*- coding: utf-8 -*-
"""
CAPER-AMC Training, Evaluation, and All-Figure Generation Script
"""

import os
import gc
import json
import math
import time
import copy
import random
import pickle
import zipfile
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Reproducibility --------------------
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -------------------- Dataset Configuration --------------------
MIN_SNR = -6
MAX_SNR = 18

# Benchmark splits for higher accuracy
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# -------------------- Training Configuration --------------------
BATCH_SIZE = 1024
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0  # Disabled for crisp logits on QAM
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.999

# -------------------- Model Configuration --------------------
D_MODEL = 128
N_HEADS = 4
ATTENTION_LAYERS_PER_ROUTE = 2
D_FF = 384
DROPOUT = 0.15
LOCAL_WINDOW = 9

ROUTING_MODE = "conditional_top1"
ROUTER_TEMP_START = 2.0
ROUTER_TEMP_END = 0.55

USE_PHYSICS_FEATURES = True
USE_SPECTRAL_CONTEXT = True
USE_PAIR_EXPERTS = True
USE_SUPERVISED_CONTRASTIVE = True

# -------------------- Loss Emphasis --------------------
LOW_SNR_FOCUS = 0.35

LOSS_WEIGHTS = {
    "coarse": 0.15,
    "snr": 0.06,
    "qam_expert": 0.12,
    "analog_expert": 0.15,
    "pair_gate": 0.03,
    "brier": 0.02,
    "load_balance": 0.02,
    "router_z": 0.001,
}

OUTPUT_DIR = Path("./caper_amc_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# -------------------- VRAM Data Loading --------------------

def decode_modulation_name(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

def load_rml2016_10a_to_vram(dataset_path, device, min_snr=-6, max_snr=18, train_ratio=0.80, val_ratio=0.10, seed=42):
    print(f"Loading dataset directly into VRAM from: {dataset_path}")
    with open(dataset_path, "rb") as file:
        raw_data = pickle.load(file, encoding="latin1")

    class_names = sorted({decode_modulation_name(mod) for mod, _ in raw_data.keys()})
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    all_snrs = sorted({int(snr) for _, snr in raw_data.keys()})
    selected_snrs = [snr for snr in all_snrs if min_snr <= snr <= max_snr]

    split_storage = {
        "train": {"X": [], "y": [], "snr": []},
        "val": {"X": [], "y": [], "snr": []},
        "test": {"X": [], "y": [], "snr": []},
    }

    generator = np.random.default_rng(seed)

    for (raw_modulation, raw_snr), samples in raw_data.items():
        snr_value = int(raw_snr)
        if not (min_snr <= snr_value <= max_snr):
            continue

        class_index = class_to_index[decode_modulation_name(raw_modulation)]
        samples = np.asarray(samples, dtype=np.float32)
        indices = generator.permutation(len(samples))

        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)

        split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        for split_name, current_indices in split_indices.items():
            split_storage[split_name]["X"].append(samples[current_indices])
            split_storage[split_name]["y"].append(np.full(len(current_indices), class_index, dtype=np.int64))
            split_storage[split_name]["snr"].append(np.full(len(current_indices), snr_value, dtype=np.int16))

    result = {"class_names": class_names, "snr_values": selected_snrs}

    for split_name in ["train", "val", "test"]:
        X = np.concatenate(split_storage[split_name]["X"], axis=0)
        y = np.concatenate(split_storage[split_name]["y"], axis=0)
        snr = np.concatenate(split_storage[split_name]["snr"], axis=0)

        # Convert to tensors and push directly to GPU memory
        result[split_name] = {
            "X": torch.tensor(X, dtype=torch.float32, device=device),
            "y": torch.tensor(y, dtype=torch.long, device=device),
            "snr": torch.tensor(snr, dtype=torch.float32, device=device)
        }

    del raw_data
    gc.collect()
    torch.cuda.empty_cache()
    print("Data loaded to VRAM successfully!")
    return result

# -------------------- GPU Augmentation Functions --------------------

@torch.no_grad()
def batch_normalize_gpu(x):
    """Normalizes IQ signals across time on GPU."""
    x = x - x.mean(dim=-1, keepdim=True)
    rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True)).clamp_min(1e-6)
    return x / rms

@torch.no_grad()
def gpu_batch_augment_and_normalize(x):
    """Light augmentations (phase rotation only) to maintain high-SNR integrity."""
    B = x.shape[0]

    # 1. Random Phase Rotation (Helps generalization without destroying signal)
    rotate_mask = torch.rand(B, 1, 1, device=x.device) < 0.50
    theta = (torch.rand(B, 1, 1, device=x.device) * 2.0 - 1.0) * math.pi
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    i, q = x[:, 0:1, :], x[:, 1:2, :]
    i_rot = i * cos_t - q * sin_t
    q_rot = i * sin_t + q * cos_t
    x = torch.where(rotate_mask, torch.cat([i_rot, q_rot], dim=1), x)

    # 2. Normalize
    return batch_normalize_gpu(x)

# -------------------- Model Architecture Modules --------------------

class PhysicsFeatureFrontEnd(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        i, q = x[:, 0:1], x[:, 1:2]
        amplitude = torch.sqrt(i.square() + q.square() + self.eps)
        amp_prev = torch.cat([amplitude[:, :, :1], amplitude[:, :, :-1]], dim=-1)
        amplitude_delta = amplitude - amp_prev
        i_prev = torch.cat([i[:, :, :1], i[:, :, :-1]], dim=-1)
        q_prev = torch.cat([q[:, :, :1], q[:, :, :-1]], dim=-1)
        denom = (amplitude * amp_prev).clamp_min(self.eps)
        phase_cosine = (i * i_prev + q * q_prev) / denom
        phase_sine = (q * i_prev - i * q_prev) / denom
        return torch.cat([i, q, amplitude, amplitude_delta, phase_cosine, phase_sine], dim=1)

class SpectralContext(nn.Module):
    def __init__(self, input_length, d_model, dropout):
        super().__init__()
        number_of_bins = input_length // 2 + 1
        self.encoder = nn.Sequential(
            nn.Linear(number_of_bins, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, raw_iq):
        with torch.autocast(device_type=raw_iq.device.type, enabled=False):
            i, q = raw_iq[:, 0].float(), raw_iq[:, 1].float()
            i_spec = torch.fft.rfft(i, dim=-1, norm="ortho")
            q_spec = torch.fft.rfft(q, dim=-1, norm="ortho")
            power = i_spec.abs().square() + q_spec.abs().square()
            spectrum = torch.log1p(power)
            spectrum = (spectrum - spectrum.mean(dim=1, keepdim=True)) / spectrum.std(dim=1, keepdim=True).clamp_min(1e-5)
            context = self.encoder(spectrum)
        return context

class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool1d(x, output_size=1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.10):
        super().__init__()
        branch_ch = max(out_channels // 3, 16)
        self.branch_3 = ConvBNGELU(in_channels, branch_ch, kernel_size=3, stride=stride)
        self.branch_5 = ConvBNGELU(in_channels, branch_ch, kernel_size=5, stride=stride)
        self.branch_7 = ConvBNGELU(in_channels, branch_ch, kernel_size=7, stride=stride)
        self.fusion = nn.Sequential(
            nn.Conv1d(branch_ch * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )
        self.squeeze_excite = SqueezeExcite1D(out_channels)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        res = self.residual(x)
        feats = torch.cat([self.branch_3(x), self.branch_5(x), self.branch_7(x)], dim=1)
        feats = self.fusion(feats)
        feats = self.squeeze_excite(feats)
        return self.activation(feats + res)

class MultiHeadSDPA(nn.Module):
    def __init__(self, d_model, n_heads, dropout, causal=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.causal = causal
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        qkv = self.qkv_projection(x).view(b, l, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal)
        out = out.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.output_projection(out)

class LocalWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, window_size=9):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.window_size = window_size
        self.radius = window_size // 2
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        qkv = self.qkv_projection(x).view(b, l, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_win = F.pad(k, (0, 0, self.radius, self.radius)).unfold(dimension=2, size=self.window_size, step=1).permute(0, 1, 2, 4, 3)
        v_win = F.pad(v, (0, 0, self.radius, self.radius)).unfold(dimension=2, size=self.window_size, step=1).permute(0, 1, 2, 4, 3)

        scores = torch.einsum("bhld,bhlwd->bhlw", q, k_win) / math.sqrt(self.head_dim)
        pos = torch.arange(l, device=x.device)[:, None]
        offsets = torch.arange(-self.radius, self.radius + 1, device=x.device)[None, :]
        valid = (pos + offsets >= 0) & (pos + offsets < l)
        scores = scores.masked_fill(~valid[None, None, :, :], torch.finfo(scores.dtype).min)

        attn = F.dropout(torch.softmax(scores, dim=-1), p=self.dropout, training=self.training)
        out = torch.einsum("bhlw,bhlwd->bhld", attn, v_win).transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.output_projection(out)

class RouteTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, route_type, local_window):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if route_type == "full":
            self.attention = MultiHeadSDPA(d_model, n_heads, dropout, causal=False)
        elif route_type == "causal":
            self.attention = MultiHeadSDPA(d_model, n_heads, dropout, causal=True)
        elif route_type == "local":
            self.attention = LocalWindowAttention(d_model, n_heads, dropout, window_size=local_window)

        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.feed_forward(self.norm2(x))
        return x

class AttentionRoute(nn.Module):
    def __init__(self, route_type, d_model, n_heads, d_ff, dropout, num_layers, local_window):
        super().__init__()
        self.blocks = nn.ModuleList([RouteTransformerBlock(d_model, n_heads, d_ff, dropout, route_type, local_window) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks: x = block(x)
        return self.output_norm(x)

class DualPooling(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.fusion = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model))

    def forward(self, x):
        w = torch.softmax(self.scorer(x), dim=1)
        return self.fusion(torch.cat([torch.sum(x * w, dim=1), x.mean(dim=1)], dim=1)), w

class CAPERAMC(nn.Module):
    ROUTE_NAMES = ["full", "causal", "local"]

    def __init__(self, class_names, min_snr, max_snr, input_length=128, d_model=128, n_heads=4, route_layers=2, d_ff=384, dropout=0.15, local_window=9, routing_mode="conditional_top1"):
        super().__init__()
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.min_snr = float(min_snr)
        self.max_snr = float(max_snr)
        self.routing_mode = routing_mode
        self.class_to_index = {name: idx for idx, name in enumerate(self.class_names)}

        self.qam_indices = [self.class_to_index[name] for name in ["QAM16", "QAM64"] if name in self.class_to_index]
        self.analog_indices = [self.class_to_index[name] for name in ["AM-DSB", "WBFM"] if name in self.class_to_index]

        self.physics_frontend = PhysicsFeatureFrontEnd()
        self.spectral_context = SpectralContext(input_length, d_model, dropout)

        self.stem = ConvBNGELU(6, 64, kernel_size=7, stride=1)
        self.encoder = nn.Sequential(
            MultiScaleResidualBlock(64, d_model, stride=2, dropout=0.10),
            MultiScaleResidualBlock(d_model, d_model, stride=1, dropout=0.10),
            MultiScaleResidualBlock(d_model, d_model, stride=1, dropout=0.10),
        )

        sequence_length = input_length // 2
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.preliminary_fusion = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.coarse_classifier = nn.Linear(d_model, self.num_classes)
        self.snr_estimator = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1), nn.Tanh())

        self.router = nn.Sequential(nn.Linear(d_model + 2, d_model // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, len(self.ROUTE_NAMES)))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

        self.routes = nn.ModuleDict({r: AttentionRoute(r, d_model, n_heads, d_ff, dropout, route_layers, local_window) for r in self.ROUTE_NAMES})
        self.fusion_norm = nn.LayerNorm(d_model)
        self.temporal_pool = DualPooling(d_model, dropout)
        self.representation_fusion = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model))
        self.classifier = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, self.num_classes))
        self.embedding_head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 64))

        self.qam_expert = nn.Linear(d_model, len(self.qam_indices)) if len(self.qam_indices) == 2 else None
        self.analog_expert = nn.Linear(d_model, len(self.analog_indices)) if len(self.analog_indices) == 2 else None
        self.qam_gate_head = nn.Linear(d_model, 1) if self.qam_expert else None
        self.analog_gate_head = nn.Linear(d_model, 1) if self.analog_expert else None

    def normalized_snr_to_db(self, snr_norm):
        return (snr_norm + 1.0) * 0.5 * (self.max_snr - self.min_snr) + self.min_snr

    def forward(
        self,
        x,
        router_temperature=1.0,
        use_gumbel=True,
        routing_mode=None,
        force_route=None,
    ):
        """Forward pass with conditional, soft-all, or forced routing.

        Parameters
        ----------
        routing_mode:
            ``conditional_top1`` (default) or ``soft_all``.
        force_route:
            Optional route name: ``full``, ``causal``, or ``local``.
            This is intended for checkpoint diagnostics, not a substitute
            for independently retrained ablation experiments.
        """
        spec_ctx = self.spectral_context(x)
        tokens = self.encoder(self.stem(self.physics_frontend(x))).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, :tokens.size(1)].to(tokens.dtype)

        prelim_feats = self.preliminary_fusion(
            torch.cat([tokens.mean(dim=1), spec_ctx], dim=1)
        )
        coarse_logits = self.coarse_classifier(prelim_feats)
        coarse_probs = torch.softmax(coarse_logits, dim=1)
        uncertainty = -torch.sum(
            coarse_probs * torch.log(coarse_probs.clamp_min(1e-8)),
            dim=1,
            keepdim=True,
        ) / math.log(self.num_classes)
        snr_norm = self.snr_estimator(prelim_feats)

        router_logits = self.router(
            torch.cat([prelim_feats, snr_norm, uncertainty], dim=1)
        )
        routing_probs = torch.softmax(router_logits, dim=1)
        active_mode = routing_mode or self.routing_mode

        if force_route is not None:
            if force_route not in self.ROUTE_NAMES:
                raise ValueError(
                    f"force_route must be one of {self.ROUTE_NAMES}; "
                    f"received {force_route!r}."
                )
            route_index = self.ROUTE_NAMES.index(force_route)
            route_assign = F.one_hot(
                torch.full(
                    (tokens.size(0),),
                    route_index,
                    dtype=torch.long,
                    device=tokens.device,
                ),
                num_classes=len(self.ROUTE_NAMES),
            ).to(tokens.dtype)
            fused_tokens = self.routes[force_route](tokens)

        elif active_mode == "soft_all":
            route_assign = routing_probs.to(tokens.dtype)
            fused_tokens = torch.zeros_like(tokens)
            for route_index, route_name in enumerate(self.ROUTE_NAMES):
                route_output = self.routes[route_name](tokens)
                route_weight = routing_probs[:, route_index].view(-1, 1, 1)
                fused_tokens = fused_tokens + route_output * route_weight

        elif active_mode == "conditional_top1":
            if self.training and use_gumbel:
                route_assign = F.gumbel_softmax(
                    router_logits,
                    tau=router_temperature,
                    hard=True,
                    dim=1,
                )
            else:
                route_assign = F.one_hot(
                    routing_probs.argmax(dim=1),
                    num_classes=len(self.ROUTE_NAMES),
                ).to(tokens.dtype)

            selected_routes = route_assign.argmax(dim=1)
            fused_tokens = torch.zeros_like(tokens)

            for route_index, route_name in enumerate(self.ROUTE_NAMES):
                selected_indices = torch.nonzero(
                    selected_routes == route_index,
                    as_tuple=False,
                ).squeeze(1)
                if selected_indices.numel() == 0:
                    continue
                route_output = self.routes[route_name](
                    tokens.index_select(0, selected_indices)
                )
                route_gate = route_assign.index_select(
                    0,
                    selected_indices,
                )[:, route_index].view(-1, 1, 1)
                fused_tokens = fused_tokens.index_copy(
                    0,
                    selected_indices,
                    (route_output * route_gate).to(fused_tokens.dtype),
                )
        else:
            raise ValueError(
                "routing_mode must be 'conditional_top1' or 'soft_all'; "
                f"received {active_mode!r}."
            )

        temporal_features, pooling_weights = self.temporal_pool(
            self.fusion_norm(fused_tokens)
        )
        representation = self.representation_fusion(
            torch.cat([temporal_features, spec_ctx], dim=1)
        )
        base_logits = self.classifier(representation)
        final_logits = base_logits.clone()

        if self.qam_expert is not None:
            qam_gate = 0.5 * (
                coarse_probs[:, self.qam_indices].sum(dim=1, keepdim=True)
                + torch.sigmoid(self.qam_gate_head(prelim_feats))
            )
            qam_replacement = (
                torch.logsumexp(
                    base_logits[:, self.qam_indices],
                    dim=1,
                    keepdim=True,
                )
                + F.log_softmax(self.qam_expert(representation), dim=1)
            )
            qam_update = (
                base_logits[:, self.qam_indices] * (1.0 - qam_gate)
                + qam_replacement * qam_gate
            )
            final_logits[:, self.qam_indices] = qam_update.to(final_logits.dtype)

        if self.analog_expert is not None:
            analog_gate = 0.5 * (
                coarse_probs[:, self.analog_indices].sum(dim=1, keepdim=True)
                + torch.sigmoid(self.analog_gate_head(prelim_feats))
            )
            analog_replacement = (
                torch.logsumexp(
                    base_logits[:, self.analog_indices],
                    dim=1,
                    keepdim=True,
                )
                + F.log_softmax(self.analog_expert(representation), dim=1)
            )
            analog_update = (
                base_logits[:, self.analog_indices] * (1.0 - analog_gate)
                + analog_replacement * analog_gate
            )
            final_logits[:, self.analog_indices] = analog_update.to(
                final_logits.dtype
            )

        return {
            "logits": final_logits,
            "coarse_logits": coarse_logits,
            "snr_normalized": snr_norm,
            "router_logits": router_logits,
            "routing_probabilities": routing_probs,
            "route_assignments": route_assign,
            "uncertainty": uncertainty,
            "pooling_weights": pooling_weights,
        }

# -------------------- Loss & Metric Functions --------------------

def compute_caper_loss(model, outputs, targets, snr_db):
    per_sample_main = F.cross_entropy(outputs["logits"], targets, reduction="none", label_smoothing=LABEL_SMOOTHING)
    snr_diff = 1.0 + LOW_SNR_FOCUS * (MAX_SNR - snr_db) / (MAX_SNR - MIN_SNR)
    main_loss = torch.mean(per_sample_main * (snr_diff / snr_diff.mean().detach()))

    coarse_loss = F.cross_entropy(outputs["coarse_logits"], targets, label_smoothing=0.0)
    snr_targets = (2.0 * (snr_db - MIN_SNR) / (MAX_SNR - MIN_SNR) - 1.0).clamp(-1.0, 1.0).unsqueeze(1)
    snr_loss = F.smooth_l1_loss(outputs["snr_normalized"], snr_targets)

    prob = torch.softmax(outputs["logits"], dim=1)
    brier_loss = torch.mean(torch.sum((prob - F.one_hot(targets, num_classes=len(model.class_names)).float()).square(), dim=1))

    m_prob, m_assign = outputs["routing_probabilities"].mean(dim=0), outputs["route_assignments"].mean(dim=0)
    load_balance_loss = len(model.ROUTE_NAMES) * torch.sum(m_prob * m_assign)
    router_z_loss = torch.mean(torch.logsumexp(outputs["router_logits"], dim=1).square())

    total_loss = (main_loss + LOSS_WEIGHTS["coarse"] * coarse_loss + LOSS_WEIGHTS["snr"] * snr_loss +
                  LOSS_WEIGHTS["brier"] * brier_loss + LOSS_WEIGHTS["load_balance"] * load_balance_loss +
                  LOSS_WEIGHTS["router_z"] * router_z_loss)
    return total_loss

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for key, ema_v in self.model.state_dict().items():
            src_v = model.state_dict()[key].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(src_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(src_v)

def amp_context(): return torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()

# -------------------- VRAM Training/Eval Loops --------------------

def expected_calibration_error(probabilities, targets, number_of_bins=15):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        if not np.any(in_bin):
            continue
        ece += np.mean(in_bin) * abs(
            correctness[in_bin].mean() - confidences[in_bin].mean()
        )
    return float(ece)


def train_epoch_vram(model, data_dict, optimizer, scheduler, scaler, ema, router_temp):
    model.train()
    X, y, snr = data_dict["X"], data_dict["y"], data_dict["snr"]
    num_samples = X.size(0)
    indices = torch.randperm(num_samples, device=DEVICE)

    total_loss = 0.0
    total_correct = 0
    route_totals = torch.zeros(len(model.ROUTE_NAMES), device=DEVICE)

    for start in range(0, num_samples, BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        inputs = X[batch_indices]
        targets = y[batch_indices]
        snr_db = snr[batch_indices]

        inputs = gpu_batch_augment_and_normalize(inputs)
        optimizer.zero_grad(set_to_none=True)

        with amp_context():
            outputs = model(
                inputs,
                router_temperature=router_temp,
                use_gumbel=True,
                routing_mode="conditional_top1",
            )
            loss = compute_caper_loss(model, outputs, targets, snr_db)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        batch_count = len(batch_indices)
        total_loss += loss.item() * batch_count
        total_correct += (
            outputs["logits"].argmax(dim=1) == targets
        ).sum().item()
        route_totals += outputs["route_assignments"].detach().sum(dim=0)

    route_fractions = (route_totals / max(num_samples, 1)).detach().cpu().numpy()
    result = {
        "loss": total_loss / max(num_samples, 1),
        "accuracy": 100.0 * total_correct / max(num_samples, 1),
    }
    for route_name, route_fraction in zip(model.ROUTE_NAMES, route_fractions):
        result[f"route_{route_name}"] = float(route_fraction)
    return result


@torch.no_grad()
def evaluate_vram(
    model,
    data_dict,
    collect_outputs=False,
    routing_mode="conditional_top1",
    force_route=None,
):
    model.eval()
    X, y, snr = data_dict["X"], data_dict["y"], data_dict["snr"]
    num_samples = X.size(0)
    eval_batch_size = max(BATCH_SIZE * 2, 1)

    total_loss = 0.0
    total_correct = 0
    route_totals = torch.zeros(len(model.ROUTE_NAMES), device=DEVICE)

    all_targets = []
    all_predictions = []
    all_snrs = []
    all_probabilities = []
    all_route_probabilities = []
    all_route_assignments = []
    all_snr_predictions = []
    all_uncertainty = []

    for start in range(0, num_samples, eval_batch_size):
        inputs = X[start:start + eval_batch_size]
        targets = y[start:start + eval_batch_size]
        snr_db = snr[start:start + eval_batch_size]
        inputs = batch_normalize_gpu(inputs)

        with amp_context():
            outputs = model(
                inputs,
                use_gumbel=False,
                routing_mode=routing_mode,
                force_route=force_route,
            )
            batch_loss = F.cross_entropy(outputs["logits"], targets)

        probabilities = torch.softmax(outputs["logits"], dim=1)
        predictions = probabilities.argmax(dim=1)
        batch_count = targets.numel()

        total_loss += batch_loss.item() * batch_count
        total_correct += (predictions == targets).sum().item()
        route_totals += outputs["route_assignments"].detach().sum(dim=0)

        if collect_outputs:
            all_targets.append(targets.detach().cpu().numpy())
            all_predictions.append(predictions.detach().cpu().numpy())
            all_snrs.append(snr_db.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().float().cpu().numpy())
            all_route_probabilities.append(
                outputs["routing_probabilities"].detach().float().cpu().numpy()
            )
            all_route_assignments.append(
                outputs["route_assignments"].detach().float().cpu().numpy()
            )
            all_snr_predictions.append(
                model.normalized_snr_to_db(outputs["snr_normalized"])
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            all_uncertainty.append(
                outputs["uncertainty"].detach().float().cpu().numpy().reshape(-1)
            )

    result = {
        "loss": total_loss / max(num_samples, 1),
        "accuracy": 100.0 * total_correct / max(num_samples, 1),
    }
    route_fractions = (route_totals / max(num_samples, 1)).detach().cpu().numpy()
    for route_name, route_fraction in zip(model.ROUTE_NAMES, route_fractions):
        result[f"route_{route_name}"] = float(route_fraction)

    if not collect_outputs:
        result["macro_f1"] = float("nan")
        # Compute macro-F1 only when requested to avoid retaining all arrays.
        # Validation calls below request outputs to keep the score exact.
        return result

    targets_np = np.concatenate(all_targets)
    predictions_np = np.concatenate(all_predictions)
    snrs_np = np.concatenate(all_snrs).astype(int)
    probabilities_np = np.concatenate(all_probabilities)
    route_probabilities_np = np.concatenate(all_route_probabilities)
    route_assignments_np = np.concatenate(all_route_assignments)
    snr_predictions_np = np.concatenate(all_snr_predictions)
    uncertainty_np = np.concatenate(all_uncertainty)

    result.update(
        {
            "macro_f1": f1_score(
                targets_np,
                predictions_np,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                targets_np,
                predictions_np,
                average="weighted",
                zero_division=0,
            ),
            "ece": expected_calibration_error(probabilities_np, targets_np),
            "targets": targets_np,
            "predictions": predictions_np,
            "snrs": snrs_np,
            "probabilities": probabilities_np,
            "route_probabilities": route_probabilities_np,
            "route_assignments": route_assignments_np,
            "snr_predictions": snr_predictions_np,
            "uncertainty_values": uncertainty_np,
        }
    )
    return result


@torch.no_grad()
def measure_inference_time(
    model,
    data_dict,
    maximum_batches=50,
    routing_mode="conditional_top1",
    force_route=None,
):
    model.eval()
    X = data_dict["X"]
    batch_size = min(max(BATCH_SIZE * 2, 1), X.size(0))
    starts = list(range(0, X.size(0), batch_size))[:maximum_batches]
    if not starts:
        return float("nan")

    warmup_count = min(5, len(starts))
    for start in starts[:warmup_count]:
        inputs = batch_normalize_gpu(X[start:start + batch_size])
        with amp_context():
            model(
                inputs,
                use_gumbel=False,
                routing_mode=routing_mode,
                force_route=force_route,
            )
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    total_seconds = 0.0
    total_samples = 0
    for start in starts:
        inputs = batch_normalize_gpu(X[start:start + batch_size])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        begin = time.perf_counter()
        with amp_context():
            model(
                inputs,
                use_gumbel=False,
                routing_mode=routing_mode,
                force_route=force_route,
            )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        total_seconds += time.perf_counter() - begin
        total_samples += inputs.size(0)

    return 1000.0 * total_seconds / max(total_samples, 1)

# -------------------- Figure and Result Generation --------------------


def save_current_figure(base_path, dpi=300):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_architecture(figure_directory):
    figure_directory = Path(figure_directory)
    plt.figure(figsize=(18, 10))
    axis = plt.gca()
    axis.set_xlim(0, 18)
    axis.set_ylim(0, 10)
    axis.axis("off")

    def add_box(x, y, width, height, text, fontsize=10):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04",
            fill=False,
            linewidth=1.5,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True,
        )
        return (x, y, width, height)

    def add_arrow(start, end):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.3,
            )
        )

    add_box(0.3, 4.2, 1.6, 1.2, "I/Q input\n2 × 128")
    add_box(
        2.3,
        4.0,
        2.2,
        1.6,
        "Physics-feature front end\nI, Q, amplitude, Δamplitude,\nphase cosine, phase sine",
        fontsize=9,
    )
    add_box(
        4.9,
        4.0,
        2.3,
        1.6,
        "Multi-scale residual CNN\nkernels 3, 5, 7\nSqueeze-and-Excitation",
        fontsize=9,
    )
    add_box(2.5, 7.2, 2.3, 1.3, "Spectral context\nFFT power encoder")
    add_box(
        7.7,
        4.0,
        2.2,
        1.6,
        "Preliminary fusion\ncoarse classifier\nSNR estimator\nuncertainty",
        fontsize=9,
    )
    add_box(10.4, 4.2, 1.8, 1.2, "Conditional\nTop-1 router")
    add_box(12.8, 7.0, 2.0, 1.1, "Full attention")
    add_box(12.8, 4.6, 2.0, 1.1, "Causal attention")
    add_box(12.8, 2.2, 2.0, 1.1, "Local attention\nwindow = 9")
    add_box(15.3, 4.0, 2.1, 1.6, "Dual pooling\nattention pooling + mean")
    add_box(12.8, 0.4, 2.0, 1.1, "QAM and analog\npair experts")
    add_box(15.3, 0.4, 2.1, 1.1, "Final classifier\n11 modulations")

    add_arrow((1.9, 4.8), (2.3, 4.8))
    add_arrow((4.5, 4.8), (4.9, 4.8))
    add_arrow((7.2, 4.8), (7.7, 4.8))
    add_arrow((9.9, 4.8), (10.4, 4.8))
    add_arrow((4.8, 7.85), (8.1, 5.6))
    add_arrow((1.1, 5.4), (3.4, 7.2))
    add_arrow((12.2, 4.8), (12.8, 7.55))
    add_arrow((12.2, 4.8), (12.8, 5.15))
    add_arrow((12.2, 4.8), (12.8, 2.75))
    add_arrow((14.8, 7.55), (15.3, 5.3))
    add_arrow((14.8, 5.15), (15.3, 4.8))
    add_arrow((14.8, 2.75), (15.3, 4.3))
    add_arrow((16.35, 4.0), (16.35, 1.5))
    add_arrow((14.8, 0.95), (15.3, 0.95))
    add_arrow((9.0, 4.0), (13.8, 1.5))

    plt.title("Overall Architecture of the Proposed CAPER-AMC Framework", fontsize=16)
    save_current_figure(figure_directory / "figure_01_caper_amc_architecture")


def plot_training_figures(history_dataframe, figure_directory):
    if history_dataframe.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["train_accuracy"],
        label="Training accuracy",
    )
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["val_accuracy"],
        label="Validation accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("CAPER-AMC Training and Validation Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    save_current_figure(Path(figure_directory) / "figure_02_training_accuracy")

    plt.figure(figsize=(9, 5))
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["train_loss"],
        label="Training loss",
    )
    plt.plot(
        history_dataframe["epoch"],
        history_dataframe["val_loss"],
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("CAPER-AMC Training and Validation Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    save_current_figure(Path(figure_directory) / "figure_03_training_loss")

    route_columns = [
        column
        for column in ["route_full", "route_causal", "route_local"]
        if column in history_dataframe.columns
    ]
    if route_columns:
        plt.figure(figsize=(9, 5))
        for column in route_columns:
            plt.plot(
                history_dataframe["epoch"],
                history_dataframe[column],
                label=column.replace("route_", "").title(),
            )
        plt.xlabel("Epoch")
        plt.ylabel("Selected-sample fraction")
        plt.title("Conditional Attention-Route Usage During Training")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend()
        save_current_figure(
            Path(figure_directory) / "figure_04_route_usage_during_training"
        )


def build_result_dataframes(test_metrics, class_names, route_names):
    targets = test_metrics["targets"]
    predictions = test_metrics["predictions"]
    snrs = test_metrics["snrs"]
    route_assignments = test_metrics["route_assignments"]
    selected_routes = route_assignments.argmax(axis=1)

    precision, recall, f1_values, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    class_metrics_dataframe = pd.DataFrame(
        {
            "modulation": class_names,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_values,
            "support": support,
        }
    )

    snr_rows = []
    for snr_value in sorted(np.unique(snrs)):
        mask = snrs == snr_value
        row = {
            "snr_db": int(snr_value),
            "samples": int(mask.sum()),
            "accuracy_percent": 100.0 * accuracy_score(
                targets[mask], predictions[mask]
            ),
            "macro_f1": f1_score(
                targets[mask],
                predictions[mask],
                average="macro",
                zero_division=0,
            ),
        }
        for route_index, route_name in enumerate(route_names):
            row[f"route_{route_name}"] = float(
                np.mean(selected_routes[mask] == route_index)
            )
        snr_rows.append(row)
    snr_dataframe = pd.DataFrame(snr_rows)

    class_route_rows = []
    for class_index, class_name in enumerate(class_names):
        mask = targets == class_index
        row = {
            "modulation": class_name,
            "samples": int(mask.sum()),
        }
        for route_index, route_name in enumerate(route_names):
            row[f"route_{route_name}"] = float(
                np.mean(selected_routes[mask] == route_index)
            )
        class_route_rows.append(row)
    route_by_class_dataframe = pd.DataFrame(class_route_rows)

    confusion = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
    )
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized_confusion = confusion / np.maximum(row_sums, 1)
    confusion_dataframe = pd.DataFrame(
        normalized_confusion,
        index=class_names,
        columns=class_names,
    )

    prediction_dataframe = pd.DataFrame(
        {
            "true_index": targets,
            "true_label": [class_names[index] for index in targets],
            "predicted_index": predictions,
            "predicted_label": [class_names[index] for index in predictions],
            "snr_db": snrs,
            "confidence": test_metrics["probabilities"].max(axis=1),
            "selected_route": [route_names[index] for index in selected_routes],
            "predicted_snr_db": test_metrics["snr_predictions"],
            "uncertainty": test_metrics["uncertainty_values"],
        }
    )

    return {
        "class_metrics": class_metrics_dataframe,
        "snr_metrics": snr_dataframe,
        "route_by_class": route_by_class_dataframe,
        "confusion": confusion_dataframe,
        "predictions": prediction_dataframe,
    }


def plot_evaluation_figures(
    result_frames,
    diagnostics_dataframe,
    class_names,
    route_names,
    figure_directory,
):
    figure_directory = Path(figure_directory)
    snr_dataframe = result_frames["snr_metrics"]

    plt.figure(figsize=(9, 5))
    plt.plot(
        snr_dataframe["snr_db"],
        snr_dataframe["accuracy_percent"],
        marker="o",
    )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy (%)")
    plt.title("CAPER-AMC Classification Accuracy versus SNR")
    plt.grid(alpha=0.25)
    save_current_figure(figure_directory / "figure_05_accuracy_by_snr")

    normalized_confusion = result_frames["confusion"].to_numpy()
    plt.figure(figsize=(11, 9))
    image = plt.imshow(normalized_confusion, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(image, label="Normalized frequency")
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized Confusion Matrix of CAPER-AMC")
    for row_index in range(len(class_names)):
        for column_index in range(len(class_names)):
            value = normalized_confusion[row_index, column_index]
            if value >= 0.05:
                plt.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    save_current_figure(
        figure_directory / "figure_06_normalized_confusion_matrix"
    )

    plt.figure(figsize=(9, 5))
    for route_name in route_names:
        plt.plot(
            snr_dataframe["snr_db"],
            snr_dataframe[f"route_{route_name}"],
            marker="o",
            label=route_name.title(),
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Selected-sample fraction")
    plt.title("Conditional Attention-Route Selection versus SNR")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    save_current_figure(
        figure_directory / "figure_07_route_selection_by_snr"
    )

    route_by_class = result_frames["route_by_class"]
    x_positions = np.arange(len(route_by_class))
    bar_width = 0.8 / len(route_names)
    plt.figure(figsize=(13, 6))
    for route_index, route_name in enumerate(route_names):
        plt.bar(
            x_positions + route_index * bar_width,
            route_by_class[f"route_{route_name}"],
            width=bar_width,
            label=route_name.title(),
        )
    plt.xticks(
        x_positions + bar_width * (len(route_names) - 1) / 2,
        route_by_class["modulation"],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Selected-sample fraction")
    plt.xlabel("Modulation class")
    plt.title("Conditional Attention-Route Selection by Modulation Class")
    plt.ylim(0.0, 1.0)
    plt.legend()
    save_current_figure(
        figure_directory / "figure_08_route_selection_by_modulation"
    )

    class_metrics = result_frames["class_metrics"]
    metric_names = ["precision", "recall", "f1_score"]
    x_positions = np.arange(len(class_metrics))
    bar_width = 0.8 / len(metric_names)
    plt.figure(figsize=(13, 6))
    for metric_index, metric_name in enumerate(metric_names):
        plt.bar(
            x_positions + metric_index * bar_width,
            class_metrics[metric_name],
            width=bar_width,
            label=metric_name.replace("_", " ").title(),
        )
    plt.xticks(
        x_positions + bar_width * (len(metric_names) - 1) / 2,
        class_metrics["modulation"],
        rotation=45,
        ha="right",
    )
    plt.xlabel("Modulation class")
    plt.ylabel("Score")
    plt.title("Per-Modulation Precision, Recall, and F1-Score")
    plt.ylim(0.0, 1.05)
    plt.legend()
    save_current_figure(
        figure_directory / "figure_09_per_modulation_metrics"
    )

    if diagnostics_dataframe is not None and not diagnostics_dataframe.empty:
        plt.figure(figsize=(9, 6))
        plt.scatter(
            diagnostics_dataframe["inference_ms_per_sample"],
            diagnostics_dataframe["accuracy_percent"],
            s=80,
        )
        for _, row in diagnostics_dataframe.iterrows():
            plt.annotate(
                row["configuration"],
                (
                    row["inference_ms_per_sample"],
                    row["accuracy_percent"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        plt.xlabel("Inference time (ms per sample)")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy-Latency Trade-off Across Routing Configurations")
        plt.grid(alpha=0.25)
        save_current_figure(
            figure_directory / "figure_10_accuracy_latency_tradeoff"
        )


def create_ablation_template(output_directory):
    path = Path(output_directory) / "ablation_results.csv"
    if path.exists():
        return path
    dataframe = pd.DataFrame(
        {
            "configuration": [
                "A0 paper-style baseline",
                "A1 previous UCA-style fusion",
                "A2 conditional routing only",
                "A3 + physics features",
                "A4 + spectral context",
                "A5 + pair experts",
                "A6 complete CAPER-AMC",
            ],
            "accuracy_mean": [np.nan] * 7,
            "accuracy_std": [np.nan] * 7,
            "macro_f1_mean": [np.nan] * 7,
            "macro_f1_std": [np.nan] * 7,
            "low_snr_accuracy_mean": [np.nan] * 7,
            "low_snr_accuracy_std": [np.nan] * 7,
            "parameters_million": [np.nan] * 7,
            "latency_ms_per_sample": [np.nan] * 7,
        }
    )
    dataframe.to_csv(path, index=False)
    return path


def plot_ablation_if_available(output_directory, figure_directory):
    path = create_ablation_template(output_directory)
    dataframe = pd.read_csv(path)
    valid = dataframe.dropna(subset=["accuracy_mean"])
    if valid.empty:
        print(
            "Ablation figure skipped: fill ablation_results.csv with independently "
            "retrained A0-A6 results, then rerun with --skip-training."
        )
        return

    plt.figure(figsize=(12, 6))
    x_positions = np.arange(len(valid))
    errors = valid["accuracy_std"].fillna(0.0)
    plt.bar(
        x_positions,
        valid["accuracy_mean"],
        yerr=errors,
        capsize=4,
    )
    plt.xticks(
        x_positions,
        valid["configuration"],
        rotation=35,
        ha="right",
    )
    plt.ylabel("Accuracy (%)")
    plt.title("CAPER-AMC Ablation Study")
    save_current_figure(
        Path(figure_directory) / "figure_11_ablation_study"
    )


def save_result_tables(result_frames, output_directory):
    output_directory = Path(output_directory)
    for name, dataframe in result_frames.items():
        dataframe.to_csv(output_directory / f"{name}.csv", index=name == "confusion")


def run_routing_diagnostics(model, test_data, maximum_batches):
    configurations = [
        {
            "configuration": "Conditional Top-1",
            "routing_mode": "conditional_top1",
            "force_route": None,
        },
        {
            "configuration": "Soft All",
            "routing_mode": "soft_all",
            "force_route": None,
        },
        {
            "configuration": "Forced Full",
            "routing_mode": "conditional_top1",
            "force_route": "full",
        },
        {
            "configuration": "Forced Causal",
            "routing_mode": "conditional_top1",
            "force_route": "causal",
        },
        {
            "configuration": "Forced Local",
            "routing_mode": "conditional_top1",
            "force_route": "local",
        },
    ]

    rows = []
    for configuration in configurations:
        print(f"Evaluating diagnostic: {configuration['configuration']}")
        metrics = evaluate_vram(
            model,
            test_data,
            collect_outputs=True,
            routing_mode=configuration["routing_mode"],
            force_route=configuration["force_route"],
        )
        latency = measure_inference_time(
            model,
            test_data,
            maximum_batches=maximum_batches,
            routing_mode=configuration["routing_mode"],
            force_route=configuration["force_route"],
        )
        rows.append(
            {
                "configuration": configuration["configuration"],
                "accuracy_percent": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "ece": metrics["ece"],
                "inference_ms_per_sample": latency,
            }
        )
    return pd.DataFrame(rows)


def resolve_dataset_path(dataset_argument):
    if dataset_argument:
        candidate = Path(dataset_argument).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {candidate}")
        if candidate.suffix.lower() == ".zip":
            extract_directory = Path("./extracted_rml")
            extract_directory.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(candidate, "r") as archive:
                archive.extractall(extract_directory)
            candidates = list(extract_directory.rglob("*.pkl")) + list(
                extract_directory.rglob("*.pickle")
            )
            if not candidates:
                raise FileNotFoundError("No PKL/PICKLE dataset found inside ZIP.")
            return candidates[0]
        return candidate

    pkl_files = list(Path(".").glob("*.pkl")) + list(Path(".").glob("*.pickle"))
    if pkl_files:
        return pkl_files[0]

    zip_files = list(Path(".").glob("*.zip"))
    if zip_files:
        return resolve_dataset_path(str(zip_files[0]))

    raise FileNotFoundError(
        "Dataset not found. Supply --dataset /path/to/RML2016.10a_dict.pkl "
        "or place a PKL/PICKLE/ZIP file in the current directory."
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train CAPER-AMC and generate publication-ready result figures, "
            "tables, diagnostics, and an architecture diagram."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to RML2016.10a PKL/PICKLE or ZIP file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="caper_amc_results",
        help="Directory for checkpoint, CSV files, and figures.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to OUTPUT_DIR/caper_amc_best_vram.pth.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load the checkpoint and only regenerate evaluation outputs/figures.",
    )
    parser.add_argument(
        "--architecture-only",
        action="store_true",
        help="Generate only the architecture figure and ablation CSV template.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip soft/forced-route latency diagnostics.",
    )
    parser.add_argument(
        "--latency-batches",
        type=int,
        default=30,
        help="Maximum batches used for each latency measurement.",
    )
    return parser.parse_args()


def main():
    global OUTPUT_DIR, BATCH_SIZE, NUM_EPOCHS

    arguments = parse_arguments()
    OUTPUT_DIR = Path(arguments.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_directory = OUTPUT_DIR / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    BATCH_SIZE = int(arguments.batch_size)
    NUM_EPOCHS = int(arguments.epochs)

    set_seed(SEED)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    plot_architecture(figure_directory)
    create_ablation_template(OUTPUT_DIR)
    if arguments.architecture_only:
        print(f"Architecture figure saved in: {figure_directory.resolve()}")
        return

    dataset_path = resolve_dataset_path(arguments.dataset)
    vram_data = load_rml2016_10a_to_vram(
        dataset_path,
        DEVICE,
        min_snr=MIN_SNR,
        max_snr=MAX_SNR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )
    class_names = vram_data["class_names"]

    model = CAPERAMC(
        class_names,
        MIN_SNR,
        MAX_SNR,
        input_length=vram_data["train"]["X"].shape[-1],
        d_model=D_MODEL,
        n_heads=N_HEADS,
        route_layers=ATTENTION_LAYERS_PER_ROUTE,
        d_ff=D_FF,
        dropout=DROPOUT,
        local_window=LOCAL_WINDOW,
        routing_mode=ROUTING_MODE,
    ).to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"\nModel initialized on {DEVICE}. Total parameters: {parameter_count:,}")

    checkpoint_path = (
        Path(arguments.checkpoint)
        if arguments.checkpoint
        else OUTPUT_DIR / "caper_amc_best_vram.pth"
    )
    history_path = OUTPUT_DIR / "training_history.csv"
    history_rows = []

    if not arguments.skip_training:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        steps_per_epoch = math.ceil(
            vram_data["train"]["X"].size(0) / BATCH_SIZE
        )
        total_steps = NUM_EPOCHS * steps_per_epoch
        warmup_steps = WARMUP_EPOCHS * steps_per_epoch

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                max(
                    (step + 1) / max(warmup_steps, 1),
                    MIN_LEARNING_RATE / LEARNING_RATE,
                )
                if step < warmup_steps
                else (MIN_LEARNING_RATE / LEARNING_RATE)
                + (
                    1.0 - (MIN_LEARNING_RATE / LEARNING_RATE)
                )
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (step - warmup_steps)
                        / max(total_steps - warmup_steps, 1)
                    )
                )
            ),
        )
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        ema = ModelEMA(model, decay=EMA_DECAY)

        best_validation_score = -float("inf")
        print("\nStarting CAPER-AMC training...")
        training_start = time.time()

        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_start = time.time()
            router_temperature = ROUTER_TEMP_START * (
                ROUTER_TEMP_END / ROUTER_TEMP_START
            ) ** ((epoch - 1) / max(NUM_EPOCHS - 1, 1))

            training_metrics = train_epoch_vram(
                model,
                vram_data["train"],
                optimizer,
                scheduler,
                scaler,
                ema,
                router_temperature,
            )
            validation_metrics = evaluate_vram(
                ema.model,
                vram_data["val"],
                collect_outputs=True,
            )
            validation_score = (
                0.70 * validation_metrics["accuracy"]
                + 0.30 * 100.0 * validation_metrics["macro_f1"]
            )
            epoch_seconds = time.time() - epoch_start

            history_row = {
                "epoch": epoch,
                "train_loss": training_metrics["loss"],
                "train_accuracy": training_metrics["accuracy"],
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "val_macro_f1": validation_metrics["macro_f1"],
                "router_temperature": router_temperature,
                "route_full": training_metrics["route_full"],
                "route_causal": training_metrics["route_causal"],
                "route_local": training_metrics["route_local"],
                "epoch_seconds": epoch_seconds,
            }
            history_rows.append(history_row)
            pd.DataFrame(history_rows).to_csv(history_path, index=False)

            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
                f"Train loss: {training_metrics['loss']:.4f} | "
                f"Train acc: {training_metrics['accuracy']:.2f}% | "
                f"Val loss: {validation_metrics['loss']:.4f} | "
                f"Val acc: {validation_metrics['accuracy']:.2f}% | "
                f"Val F1: {validation_metrics['macro_f1']:.4f} | "
                f"Time: {epoch_seconds:.2f}s"
            )

            if validation_score > best_validation_score:
                best_validation_score = validation_score
                torch.save(
                    {
                        "ema_state_dict": ema.model.state_dict(),
                        "class_names": class_names,
                        "model_config": {
                            "d_model": D_MODEL,
                            "n_heads": N_HEADS,
                            "route_layers": ATTENTION_LAYERS_PER_ROUTE,
                            "d_ff": D_FF,
                            "dropout": DROPOUT,
                            "local_window": LOCAL_WINDOW,
                        },
                    },
                    checkpoint_path,
                )

        print(
            f"\nTraining completed in "
            f"{(time.time() - training_start) / 60.0:.2f} minutes."
        )
    elif history_path.exists():
        history_rows = pd.read_csv(history_path).to_dict("records")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run without "
            "--skip-training or provide --checkpoint."
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["ema_state_dict"])

    print("\nRunning detailed final evaluation...")
    test_metrics = evaluate_vram(
        model,
        vram_data["test"],
        collect_outputs=True,
    )
    result_frames = build_result_dataframes(
        test_metrics,
        class_names,
        model.ROUTE_NAMES,
    )
    save_result_tables(result_frames, OUTPUT_DIR)

    diagnostics_dataframe = None
    if not arguments.skip_diagnostics:
        diagnostics_dataframe = run_routing_diagnostics(
            model,
            vram_data["test"],
            maximum_batches=max(arguments.latency_batches, 1),
        )
        diagnostics_dataframe.to_csv(
            OUTPUT_DIR / "routing_mode_diagnostics.csv",
            index=False,
        )

    history_dataframe = (
        pd.DataFrame(history_rows)
        if history_rows
        else pd.DataFrame()
    )
    plot_training_figures(history_dataframe, figure_directory)
    plot_evaluation_figures(
        result_frames,
        diagnostics_dataframe,
        class_names,
        model.ROUTE_NAMES,
        figure_directory,
    )
    plot_ablation_if_available(OUTPUT_DIR, figure_directory)

    summary = {
        "dataset": str(dataset_path),
        "device": str(DEVICE),
        "parameter_count": int(parameter_count),
        "test_accuracy_percent": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_ece": float(test_metrics["ece"]),
        "figures_directory": str(figure_directory.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "diagnostics_note": (
            "Forced-route and soft-all values use one conditionally trained "
            "checkpoint. Publication-grade ablations require independent "
            "retraining with identical splits and multiple seeds."
        ),
    }
    with open(OUTPUT_DIR / "results_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n================ Final Test Results ================")
    print(f"Overall test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Overall test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Expected calibration error: {test_metrics['ece']:.4f}")
    print(f"All results saved to: {OUTPUT_DIR.resolve()}")
    print(f"All figures saved to: {figure_directory.resolve()}")


if __name__ == "__main__":
    main()
