
# vit_fashionmnist_partB.py (FIXED)
# Mini ViT implementation for Fashion-MNIST (Part B of assignment)
# Author: Pawan Sharma (Roll: 22053329) — hyperparameters derived from roll number.
# Framework: PyTorch (no pretrained ViT; patch embedding + attention implemented manually).
# Note: Allows hidden_dim NOT divisible by num_heads by expanding to head_dim*num_heads and projecting back.

import os
import math
import csv
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

import torchvision
from torchvision import datasets, transforms

# ==============================
# Reproducibility
# ==============================
def set_seed(seed: int = 29):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================
# Data pipeline (Fashion-MNIST)
#  - Pad 28x28 to 30x30 so 10x10 patches tile perfectly (3x3 = 9 patches)
# ==============================
def get_dataloaders(batch_size=128, seed=29, num_workers=2):
    set_seed(seed)
    normalize = transforms.Normalize((0.2860,), (0.3530,))  # standard stats for Fashion-MNIST
    train_tfms = transforms.Compose([
        transforms.Pad(1, fill=0),
        transforms.CenterCrop(30),
        transforms.ToTensor(),
        normalize
    ])
    test_tfms = transforms.Compose([
        transforms.Pad(1, fill=0),
        transforms.CenterCrop(30),
        transforms.ToTensor(),
        normalize
    ])

    root = "./data"
    train_full = datasets.FashionMNIST(root=root, train=True, download=True, transform=train_tfms)
    test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=test_tfms)

    # Split train into train/val = 54k/6k
    val_size = 6000
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ==============================
# Vision Transformer components
# ==============================

class PatchEmbed(nn.Module):
    """
    Manual patch embedding using nn.Unfold.
    Input: (B, C=1, H=30, W=30), patch_size=10 -> 9 patches.
    Each patch is flattened (C * p * p) and projected to hidden_dim.
    """
    def __init__(self, img_size=30, patch_size=10, in_chans=1, hidden_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.proj = nn.Linear(in_chans * patch_size * patch_size, hidden_dim)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        patches = self.unfold(x)  # (B, C*patch_size*patch_size, num_patches)
        patches = patches.transpose(1, 2)  # (B, num_patches, C*patch_size*patch_size)
        embeddings = self.proj(patches)    # (B, num_patches, hidden_dim)
        return embeddings

class MultiHeadSelfAttention(nn.Module):
    """
    Manual multi-head self-attention that supports hidden_dim not divisible by num_heads.
    We choose head_dim = ceil(hidden_dim / num_heads) and project back to hidden_dim.
    """
    def __init__(self, hidden_dim=256, num_heads=6, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = math.ceil(hidden_dim / num_heads)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        # project to qkv of size inner_dim each
        self.qkv = nn.Linear(hidden_dim, self.inner_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        # project back to hidden_dim
        self.proj = nn.Linear(self.inner_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*inner_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)  # (B, N, inner_dim)
        out = self.proj(out)   # (B, N, hidden_dim)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn
        return out

class MLP(nn.Module):
    def __init__(self, hidden_dim=256, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, hidden_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=6, mlp_ratio=4.0, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, drop)

    def forward(self, x, return_attn=False):
        if return_attn:
            y, attn = self.attn(self.norm1(x), return_attn=True)
            x = x + y
            x = x + self.mlp(self.norm2(x))
            return x, attn
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

class ViT(nn.Module):
    def __init__(self, img_size=30, patch_size=10, in_chans=1, num_classes=10,
                 hidden_dim=256, depth=6, num_heads=6, mlp_ratio=4.0, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, hidden_dim)
        self.num_patches = self.patch_embed.num_patches
        self.hidden_dim = hidden_dim

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, hidden_dim))
        self.pos_drop = nn.Dropout(drop)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Classifier head
        self.head = nn.Linear(hidden_dim, num_classes)

        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_last_attn=False):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, hidden_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1+num_patches, hidden_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        last_attn = None
        for i, blk in enumerate(self.blocks):
            if return_last_attn and i == len(self.blocks) - 1:
                x, last_attn = blk(x, return_attn=True)
            else:
                x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, hidden_dim)
        logits = self.head(cls_out)
        if return_last_attn:
            return logits, last_attn
        return logits

# ==============================
# Training utilities
# ==============================

class WarmupCosine:
    """
    Linear warmup to max_lr over warmup_steps, then cosine decay to min_lr.
    """
    def __init__(self, optimizer, max_lr, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            lr = self.max_lr * self._step / self.warmup_steps
        else:
            progress = (self._step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    acc = (preds == targets).float().mean().item()
    return acc

def train_one_epoch(model, loader, optimizer, device, scaler=None, scheduler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(logits, targets) * bs
        n += bs

    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy_from_logits(logits, targets) * bs
        n += bs
    return running_loss / n, running_acc / n

# ==============================
# Main training (Part B)
# ==============================
def main():
    # --------------------------
    # Roll-number-based params
    # --------------------------
    seed = 29
    hidden_dim = 128 + (seed % 5) * 32     # 256
    num_heads = 4 + (seed % 3)             # 6
    patch_size = 8 + (seed % 4) * 2        # 10
    epochs = 10 + (seed % 5)               # 14

    # Fixed choices
    img_size = 30          # because we pad 28->30
    depth = 6
    batch_size = 128
    lr = 3e-4
    weight_decay = 5e-2
    mlp_ratio = 4.0
    drop = 0.1
    attn_drop = 0.1

    out_dir = Path("outputs_partB")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs.csv"
    ckpt_path = out_dir / "model_best.pt"

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {seed}")
    print(f"Hyperparams -> hidden_dim={hidden_dim}, num_heads={num_heads}, patch_size={patch_size}, epochs={epochs}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, seed=seed)

    # Model
    model = ViT(img_size=img_size, patch_size=patch_size, in_chans=1, num_classes=10,
                hidden_dim=hidden_dim, depth=depth, num_heads=num_heads,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(0.1 * total_steps))  # 10% warmup
    scheduler = WarmupCosine(optimizer, max_lr=lr, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=1e-6)

    # AMP
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # CSV logging
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, device)

        # current LR (from optimizer)
        cur_lr = optimizer.param_groups[0]["lr"]

        # Logging
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{cur_lr:.8f}"])

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "config": {
                    "img_size": img_size,
                    "patch_size": patch_size,
                    "hidden_dim": hidden_dim,
                    "depth": depth,
                    "num_heads": num_heads,
                    "mlp_ratio": mlp_ratio,
                    "drop": drop,
                    "attn_drop": attn_drop
                }
            }, ckpt_path)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs} | {dt:.1f}s | "
              f"train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"val: loss={val_loss:.4f}, acc={val_acc:.4f} | best_val_acc={best_val_acc:.4f} | lr={cur_lr:.2e}")

    total_dt = time.time() - start_time
    print(f"Training complete in {total_dt/60:.1f} min. Best val acc: {best_val_acc:.4f}")
    print(f"Logs saved to: {log_path}")
    print(f"Best checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
