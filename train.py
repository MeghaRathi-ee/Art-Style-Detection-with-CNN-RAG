"""
train.py
========
Fine-tunes ArtStyleClassifier on a WikiArt-style folder dataset.

Expected data layout (ImageFolder format):
    data/wikiart/
        Impressionism/
            monet_001.jpg
            renoir_002.jpg
            ...
        Baroque/
            caravaggio_001.jpg
            ...
        ...

Download WikiArt via:
    pip install kaggle
    kaggle datasets download -d steubk/wikiart
    unzip wikiart.zip -d data/wikiart

Training strategy (two-phase):
  Phase 1 — Warm-up (5 epochs):
      Only the new classification head is trained (backbone frozen).
      This prevents the high LR from destroying pretrained features.
  Phase 2 — Fine-tune (remaining epochs):
      All layers are unfrozen and trained with a low LR.

Usage:
    python train.py --data_dir data/wikiart --epochs 20 --batch_size 32
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from art_classifier import ArtStyleClassifier, ART_STYLES, get_transforms


# ── Helpers ────────────────────────────────────────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            total_acc += accuracy(logits, labels) * len(labels)
            n += len(labels)
    return total_loss / n, total_acc / n


# ── Training loop ──────────────────────────────────────────────────────────────
def train_model(
    data_dir: str,
    save_path: str = "art_classifier.pth",
    epochs: int = 20,
    batch_size: int = 32,
    val_split: float = 0.15,
    warmup_epochs: int = 5,
    head_lr: float = 1e-3,
    finetune_lr: float = 3e-5,
    num_workers: int = 4,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_tf = get_transforms(train=True)
    val_tf = get_transforms(train=False)

    full_dataset = ImageFolder(root=data_dir, transform=train_tf)
    print(f"[Train] Classes found: {full_dataset.classes}")
    print(f"[Train] Total images:  {len(full_dataset)}")

    # Verify classes match our expected ART_STYLES
    found = set(full_dataset.classes)
    expected = set(ART_STYLES)
    missing = expected - found
    extra = found - expected
    if missing:
        print(f"[Warn]  Missing classes: {missing}")
    if extra:
        print(f"[Warn]  Extra classes (will be ignored by model head): {extra}")

    num_classes = len(full_dataset.classes)

    # Train / val split
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # Apply val transforms to val set
    val_ds.dataset = ImageFolder(root=data_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ArtStyleClassifier(num_classes=num_classes, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_epoch = 0

    # ── Phase 1: warm-up (head only) ──────────────────────────────────────────
    print(f"\n[Train] Phase 1: warming up classifier head for {warmup_epochs} epochs...")
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=head_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs)

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        t0 = time.time()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{warmup_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_acc += accuracy(logits, labels) * len(labels)

        scheduler.step()
        n = len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"  Epoch {epoch:2d} | "
            f"train loss {train_loss/n:.4f} acc {train_acc/n:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"{time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    # ── Phase 2: full fine-tune ────────────────────────────────────────────────
    remaining = epochs - warmup_epochs
    if remaining > 0:
        print(f"\n[Train] Phase 2: fine-tuning all layers for {remaining} epochs...")
        model.unfreeze_all()
        optimizer = AdamW(model.parameters(), lr=finetune_lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

        for epoch in range(warmup_epochs + 1, epochs + 1):
            model.train()
            t0 = time.time()
            train_loss, train_acc = 0.0, 0.0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(labels)
                train_acc += accuracy(logits, labels) * len(labels)

            scheduler.step()
            n = len(train_loader.dataset)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            print(
                f"  Epoch {epoch:2d} | "
                f"train loss {train_loss/n:.4f} acc {train_acc/n:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
                f"{time.time()-t0:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)

    print(f"\n[Train] Best val accuracy: {best_val_acc:.3f} at epoch {best_epoch}")
    print(f"[Train] Model saved to: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the art style CNN")
    parser.add_argument("--data_dir", type=str, default="data/wikiart",
                        help="Path to dataset root (ImageFolder format)")
    parser.add_argument("--save_path", type=str, default="art_classifier.pth",
                        help="Where to save the best model weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
    )