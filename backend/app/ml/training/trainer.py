import os
import time
import random
from pathlib import Path
from datetime import datetime, timezone
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
import numpy as np
from typing import Dict, Any, List, Optional, Callable

from app.ml.models.factory import build_model
from app.ml.losses.thermal_loss import CompositeThermalLoss
from app.services.preprocessing import (
    load_thermal_image,
    load_rgb_image,
    match_aspect_and_crop,
    downsample_thermal,
    extract_paired_patch,
    apply_paired_augmentation,
)
from app.services.metrics import calculate_psnr, calculate_ssim
from app.core.config import settings
from app.core.logging import logger

class PairedThermalDataset(TorchDataset):
    """
    PyTorch Dataset for paired RGB and thermal training.
    """
    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        root_dir: str,
        scale_factor: int = 8,
        patch_size: int = 64,
        is_training: bool = True,
        max_samples: Optional[int] = None
    ):
        self.pairs = pairs[:max_samples] if max_samples else pairs
        self.root_dir = Path(root_dir)
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.is_training = is_training

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]
        t_path = str(self.root_dir / item["thermal_path"])
        t_hr = load_thermal_image(t_path)

        rgb_path = item.get("rgb_path")
        if rgb_path:
            r_path = str(self.root_dir / rgb_path)
            rgb_raw = load_rgb_image(r_path)
            rgb_hr, _ = match_aspect_and_crop(rgb_raw, t_hr.shape[1], t_hr.shape[0])
        else:
            # If no RGB guidance image is present in dataset, synthesize structural guidance from thermal
            rgb_hr = np.repeat(t_hr[:, :, np.newaxis], 3, axis=2)

        if self.is_training:
            # Extract random patch
            t_lr, rgb_patch, t_hr_patch = extract_paired_patch(
                t_hr, rgb_hr, patch_size_hr=self.patch_size, scale_factor=self.scale_factor
            )
            # Synchronized data augmentation
            t_lr, rgb_patch, t_hr_patch = apply_paired_augmentation(t_lr, rgb_patch, t_hr_patch)
        else:
            # Validation: use downsampled full or center crop
            t_hr_patch = t_hr
            rgb_patch = rgb_hr
            t_lr = downsample_thermal(t_hr, scale_factor=self.scale_factor)

        # Convert to tensors
        t_lr_t = torch.from_numpy(t_lr).unsqueeze(0).float()
        rgb_t = torch.from_numpy(rgb_patch).permute(2, 0, 1).float()
        t_hr_t = torch.from_numpy(t_hr_patch).unsqueeze(0).float()

        return t_lr_t, rgb_t, t_hr_t

class TrainingSession:
    def __init__(
        self,
        run_id: str,
        architecture: str,
        scale_factor: int,
        hyperparameters: Dict[str, Any],
        train_pairs: List[Dict[str, Any]],
        val_pairs: List[Dict[str, Any]],
        dataset_root: str,
        device: str = "auto",
        seed: int = 42,
        on_epoch_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.run_id = run_id
        self.architecture = architecture
        self.scale_factor = scale_factor
        self.hp = hyperparameters
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.dataset_root = dataset_root
        self.seed = seed
        self.on_epoch_complete = on_epoch_complete
        self.is_cancelled = False

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    def cancel(self):
        self.is_cancelled = True
        logger.info(f"Training run {self.run_id} marked as cancelled.")

    def run(self) -> Dict[str, Any]:
        logger.info(f"Starting training run {self.run_id} for {self.architecture} on {self.device}")
        
        # Instantiate model
        model = build_model(architecture=self.architecture, scale_factor=self.scale_factor)
        model.to(self.device)

        # Setup loss function
        criterion = CompositeThermalLoss(
            scale_factor=self.scale_factor,
            w_l1=self.hp.get("loss_reconstruction_weight", 1.0),
            w_grad=self.hp.get("loss_gradient_weight", 0.1),
            w_consist=self.hp.get("loss_thermal_consistency_weight", 0.05),
            use_grad=self.hp.get("use_gradient_loss", True),
            use_consist=self.hp.get("use_thermal_consistency_loss", True),
        ).to(self.device)

        # Optimizer
        lr = self.hp.get("learning_rate", 1e-4)
        opt_name = self.hp.get("optimizer", "adamw").lower()
        if opt_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hp.get("total_epochs", 10), eta_min=1e-6
        )

        # Create datasets & dataloaders
        batch_size = self.hp.get("batch_size", 8)
        patch_size = self.hp.get("patch_size", 64)

        train_ds = PairedThermalDataset(
            self.train_pairs, self.dataset_root,
            scale_factor=self.scale_factor, patch_size=patch_size, is_training=True
        )
        val_ds = PairedThermalDataset(
            self.val_pairs, self.dataset_root,
            scale_factor=self.scale_factor, patch_size=patch_size, is_training=False,
            max_samples=20  # Limit validation set for responsive evaluations
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        total_epochs = self.hp.get("total_epochs", 10)
        best_psnr = -1.0
        best_checkpoint_path = None
        metrics_history = []
        start_time = time.time()

        for epoch in range(1, total_epochs + 1):
            if self.is_cancelled:
                logger.info(f"Run {self.run_id} cancelled by user at epoch {epoch}")
                return {"status": "cancelled", "best_checkpoint": best_checkpoint_path, "metrics_history": metrics_history}

            # Training Phase
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for t_lr, rgb, t_hr in train_loader:
                t_lr = t_lr.to(self.device)
                rgb = rgb.to(self.device)
                t_hr = t_hr.to(self.device)

                optimizer.zero_grad()

                if self.architecture == "thermal_cnn_baseline":
                    pred = model(t_lr)
                elif self.architecture == "rgb_fusion":
                    pred = model(t_lr, rgb)
                elif self.architecture == "reliability_aware_fusion":
                    pred, _ = model(t_lr, rgb, return_reliability=True)

                loss, loss_dict = criterion(pred, t_hr, t_lr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss_dict["loss_total"]
                num_batches += 1

            avg_train_loss = round(epoch_loss / max(num_batches, 1), 4)
            scheduler.step()
            curr_lr = float(optimizer.param_groups[0]["lr"])

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_psnrs = []
            val_ssims = []

            with torch.no_grad():
                for t_lr, rgb, t_hr in val_loader:
                    t_lr = t_lr.to(self.device)
                    rgb = rgb.to(self.device)
                    t_hr = t_hr.to(self.device)

                    if self.architecture == "thermal_cnn_baseline":
                        pred = model(t_lr)
                    elif self.architecture == "rgb_fusion":
                        pred = model(t_lr, rgb)
                    elif self.architecture == "reliability_aware_fusion":
                        pred = model(t_lr, rgb, return_reliability=False)

                    v_loss, v_dict = criterion(pred, t_hr, t_lr)
                    val_loss += v_dict["loss_total"]

                    p_np = pred.squeeze().cpu().numpy()
                    t_np = t_hr.squeeze().cpu().numpy()
                    val_psnrs.append(calculate_psnr(p_np, t_np))
                    val_ssims.append(calculate_ssim(p_np, t_np))

            avg_val_loss = round(val_loss / max(len(val_loader), 1), 4)
            avg_psnr = round(float(np.mean(val_psnrs)), 2) if val_psnrs else 0.0
            avg_ssim = round(float(np.mean(val_ssims)), 4) if val_ssims else 0.0

            epoch_metric = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "lr": curr_lr,
                "elapsed_seconds": round(time.time() - start_time, 1)
            }
            metrics_history.append(epoch_metric)

            # Checkpoint saving
            ckpt_dir = settings.CHECKPOINTS_DIR / self.run_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_checkpoint_path = str(ckpt_dir / "best_model.pth")
                torch.save({
                    "architecture": self.architecture,
                    "scale_factor": self.scale_factor,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_psnr": best_psnr,
                    "best_ssim": avg_ssim,
                    "hyperparameters": self.hp,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }, best_checkpoint_path)

            # Always save latest checkpoint
            torch.save({
                "architecture": self.architecture,
                "scale_factor": self.scale_factor,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "hyperparameters": self.hp,
                "created_at": datetime.now(timezone.utc).isoformat()
            }, str(ckpt_dir / "latest_model.pth"))

            if self.on_epoch_complete:
                self.on_epoch_complete(epoch_metric)

            logger.info(
                f"Epoch [{epoch}/{total_epochs}] "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"PSNR: {avg_psnr}dB | SSIM: {avg_ssim:.4f}"
            )

        return {
            "status": "completed",
            "best_psnr": best_psnr,
            "best_checkpoint": best_checkpoint_path,
            "metrics_history": metrics_history,
            "elapsed_seconds": round(time.time() - start_time, 1)
        }
