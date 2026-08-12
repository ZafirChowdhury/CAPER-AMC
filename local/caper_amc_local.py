# -*- coding: utf-8 -*-
"""
CAPER-AMC VRAM-Optimized Script (RTX 3060 Max Throughput)
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
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

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

    def forward(self, x, router_temperature=1.0, use_gumbel=True):
        spec_ctx = self.spectral_context(x)
        tokens = self.encoder(self.stem(self.physics_frontend(x))).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, :tokens.size(1)].to(tokens.dtype)

        prelim_feats = self.preliminary_fusion(torch.cat([tokens.mean(dim=1), spec_ctx], dim=1))
        coarse_logits = self.coarse_classifier(prelim_feats)
        coarse_probs = torch.softmax(coarse_logits, dim=1)
        uncertainty = -torch.sum(coarse_probs * torch.log(coarse_probs.clamp_min(1e-8)), dim=1, keepdim=True) / math.log(self.num_classes)
        snr_norm = self.snr_estimator(prelim_feats)

        router_logits = self.router(torch.cat([prelim_feats, snr_norm, uncertainty], dim=1))
        routing_probs = torch.softmax(router_logits, dim=1)

        if self.training and use_gumbel:
            route_assign = F.gumbel_softmax(router_logits, tau=router_temperature, hard=True, dim=1)
        else:
            route_assign = F.one_hot(routing_probs.argmax(dim=1), num_classes=len(self.ROUTE_NAMES)).to(tokens.dtype)

        sel_routes = route_assign.argmax(dim=1)
        fused_tokens = torch.zeros_like(tokens)

        for r_idx, r_name in enumerate(self.ROUTE_NAMES):
            s_indices = torch.nonzero(sel_routes == r_idx, as_tuple=False).squeeze(1)
            if s_indices.numel() == 0: continue
            r_out = self.routes[r_name](tokens.index_select(0, s_indices))
            s_gate = route_assign.index_select(0, s_indices)[:, r_idx].view(-1, 1, 1)
            fused_tokens = fused_tokens.index_copy(0, s_indices, (r_out * s_gate).to(fused_tokens.dtype))

        temp_feats, pool_w = self.temporal_pool(self.fusion_norm(fused_tokens))
        rep = self.representation_fusion(torch.cat([temp_feats, spec_ctx], dim=1))
        base_logits = self.classifier(rep)
        final_logits = base_logits.clone()

        if self.qam_expert:
                    q_gate = 0.5 * (coarse_probs[:, self.qam_indices].sum(dim=1, keepdim=True) + torch.sigmoid(self.qam_gate_head(prelim_feats)))
                    q_repl = torch.logsumexp(base_logits[:, self.qam_indices], dim=1, keepdim=True) + F.log_softmax(self.qam_expert(rep), dim=1)
                    q_update = (base_logits[:, self.qam_indices] * (1.0 - q_gate) + q_repl * q_gate)
                    final_logits[:, self.qam_indices] = q_update.to(final_logits.dtype)

        if self.analog_expert:
            a_gate = 0.5 * (coarse_probs[:, self.analog_indices].sum(dim=1, keepdim=True) + torch.sigmoid(self.analog_gate_head(prelim_feats)))
            a_repl = torch.logsumexp(base_logits[:, self.analog_indices], dim=1, keepdim=True) + F.log_softmax(self.analog_expert(rep), dim=1)
            a_update = (base_logits[:, self.analog_indices] * (1.0 - a_gate) + a_repl * a_gate)
            final_logits[:, self.analog_indices] = a_update.to(final_logits.dtype)

        return {
            "logits": final_logits, "coarse_logits": coarse_logits, "snr_normalized": snr_norm,
            "router_logits": router_logits, "routing_probabilities": routing_probs, "route_assignments": route_assign
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

def train_epoch_vram(model, data_dict, optimizer, scheduler, scaler, ema, router_temp):
    model.train()
    X, y, snr = data_dict["X"], data_dict["y"], data_dict["snr"]
    num_samples = X.size(0)

    # Shuffle entirely on GPU
    indices = torch.randperm(num_samples, device=DEVICE)

    total_loss, total_correct = 0.0, 0

    for i in range(0, num_samples, BATCH_SIZE):
        batch_idx = indices[i:i + BATCH_SIZE]
        inputs, targets, snr_db = X[batch_idx], y[batch_idx], snr[batch_idx]

        inputs = gpu_batch_augment_and_normalize(inputs)
        optimizer.zero_grad(set_to_none=True)

        with amp_context():
            outputs = model(inputs, router_temperature=router_temp, use_gumbel=True)
            loss = compute_caper_loss(model, outputs, targets, snr_db)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        total_loss += loss.item() * len(batch_idx)
        total_correct += (outputs["logits"].argmax(dim=1) == targets).sum().item()

    return {"loss": total_loss / num_samples, "accuracy": 100.0 * total_correct / num_samples}

@torch.no_grad()
def evaluate_vram(model, data_dict):
    model.eval()
    X, y, snr = data_dict["X"], data_dict["y"], data_dict["snr"]
    num_samples = X.size(0)

    total_correct = 0
    all_targets, all_preds = [], []

    # Dictionary to hold accuracy metrics for each SNR
    snr_stats = {int(s): {"correct": 0, "total": 0} for s in torch.unique(snr).cpu().numpy()}

    # Evaluate in batches to avoid VRAM OOM during forward passes
    eval_batch_size = BATCH_SIZE * 2
    for i in range(0, num_samples, eval_batch_size):
        inputs, targets, snr_db = X[i:i + eval_batch_size], y[i:i + eval_batch_size], snr[i:i + eval_batch_size]

        inputs = batch_normalize_gpu(inputs)
        with amp_context():
            outputs = model(inputs, use_gumbel=False)

        preds = outputs["logits"].argmax(dim=1)
        total_correct += (preds == targets).sum().item()

        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        # Populate per-SNR stats
        for p, t, s in zip(preds, targets, snr_db):
            s_val = int(s.item())
            snr_stats[s_val]["total"] += 1
            if p == t:
                snr_stats[s_val]["correct"] += 1

    targets_np = np.concatenate(all_targets)
    preds_np = np.concatenate(all_preds)

    snr_accuracy = {k: (v["correct"] / v["total"]) * 100 if v["total"] > 0 else 0 for k, v in snr_stats.items()}

    return {
        "accuracy": 100.0 * total_correct / num_samples,
        "macro_f1": f1_score(targets_np, preds_np, average="macro", zero_division=0),
        "snr_accuracy": snr_accuracy
    }

# -------------------- Main Execution --------------------

if __name__ == "__main__":
    set_seed(SEED)
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    pkl_files = list(Path(".").glob("*.pkl")) + list(Path(".").glob("*.pickle"))
    zip_files = list(Path(".").glob("*.zip"))

    if pkl_files: dataset_path = pkl_files[0]
    elif zip_files:
        extract_dir = Path("./extracted_rml")
        with zipfile.ZipFile(zip_files[0], "r") as archive: archive.extractall(extract_dir)
        dataset_path = list(extract_dir.rglob("*.pkl"))[0]
    else: raise FileNotFoundError("Dataset not found.")

    # Load everything directly to GPU VRAM
    vram_data = load_rml2016_10a_to_vram(dataset_path, DEVICE, min_snr=MIN_SNR, max_snr=MAX_SNR, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED)
    CLASS_NAMES = vram_data["class_names"]

    model = CAPERAMC(CLASS_NAMES, MIN_SNR, MAX_SNR).to(DEVICE)
    print(f"\nModel initialized on {DEVICE}. Total Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(vram_data["train"]["X"].size(0) / BATCH_SIZE)
    total_steps, warmup_steps = NUM_EPOCHS * steps_per_epoch, WARMUP_EPOCHS * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max((step + 1) / max(warmup_steps, 1), MIN_LEARNING_RATE / LEARNING_RATE)
        if step < warmup_steps else
        (MIN_LEARNING_RATE / LEARNING_RATE) + (1.0 - (MIN_LEARNING_RATE / LEARNING_RATE)) * 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)))
    )

    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    ema = ModelEMA(model, decay=EMA_DECAY)

    best_val_score = -float("inf")
    best_model_path = OUTPUT_DIR / "caper_amc_best_vram.pth"

    print("\nStarting Max-Throughput VRAM Training...")
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        router_temp = ROUTER_TEMP_START * (ROUTER_TEMP_END / ROUTER_TEMP_START) ** ((epoch - 1) / max(NUM_EPOCHS - 1, 1))

        train_m = train_epoch_vram(model, vram_data["train"], optimizer, scheduler, scaler, ema, router_temp)
        val_m = evaluate_vram(ema.model, vram_data["val"])

        val_score = 0.70 * val_m["accuracy"] + 0.30 * (100.0 * val_m["macro_f1"])
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train Acc: {train_m['accuracy']:.2f}% | Val Acc: {val_m['accuracy']:.2f}% | Val F1: {val_m['macro_f1']:.4f} | Time: {epoch_time:.2f}s")

        if val_score > best_val_score:
            best_val_score = val_score
            torch.save({"ema_state_dict": ema.model.state_dict(), "class_names": CLASS_NAMES}, best_model_path)

    print(f"\nTraining completed in {(time.time() - start_time) / 60.0:.2f} minutes.")

    # Final Evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE)["ema_state_dict"])
    test_m = evaluate_vram(model, vram_data["test"])

    print(f"\n================ Final Test Results ================")
    print(f"Overall Test Accuracy: {test_m['accuracy']:.2f}%")
    print(f"Overall Test Macro F1: {test_m['macro_f1']:.4f}\n")

    print("--- Accuracy by SNR ---")
    for snr, acc in sorted(test_m['snr_accuracy'].items()):
        print(f"SNR {snr:3d} dB : {acc:6.2f}%")
