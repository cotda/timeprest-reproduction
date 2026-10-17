
import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# CONFIG - GIỮ GIỐNG 1 GPU
# ============================================================
SEED = 42
EPOCHS = 20
LOCAL_BATCH_SIZE = 32       # 32 x 2 GPU = global batch 64
LR = 1e-4


class NutritionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        calories = torch.tensor(
            row["calories"],
            dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return image, calories


def evaluate(model, loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for images, calories in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            calories = calories.to(
                device,
                non_blocking=True
            )

            outputs = model(images).squeeze(1)

            preds.extend(
                outputs.cpu().numpy()
            )

            targets.extend(
                calories.cpu().numpy()
            )

    preds = np.array(preds)
    targets = np.array(targets)

    mae = np.mean(
        np.abs(preds - targets)
    )

    rmse = np.sqrt(
        np.mean(
            (preds - targets) ** 2
        )
    )

    return mae, rmse


def main():

    # ========================================================
    # DDP INIT
    # ========================================================
    dist.init_process_group(
        backend="nccl"
    )

    rank = dist.get_rank()
    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}"
    )

    # Seed
    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


    # ========================================================
    # DATA
    # ========================================================
    train_df = pd.read_csv(
        "/kaggle/working/train.csv"
    )

    val_df = pd.read_csv(
        "/kaggle/working/val.csv"
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = NutritionDataset(
        train_df,
        train_transform
    )

    val_dataset = NutritionDataset(
        val_df,
        val_transform
    )


    # ========================================================
    # DISTRIBUTED SAMPLER
    # ========================================================
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=LOCAL_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    # Validation chỉ rank 0
    if rank == 0:

        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


    # ========================================================
    # RESNET50
    # ========================================================
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )

    model = model.to(device)

    model = DDP(
        model,
        device_ids=[local_rank]
    )

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )


    # ========================================================
    # TRAIN
    # ========================================================
    total_train_time = 0.0
    best_mae = float("inf")

    final_mae = None
    final_rmse = None

    history = []

    for epoch in range(EPOCHS):

        train_sampler.set_epoch(epoch)

        model.train()

        # ----------------------------
        # START TRAIN TIMER
        # ----------------------------
        dist.barrier()
        torch.cuda.synchronize()

        start = time.time()

        total_loss = 0.0
        total_samples = 0

        for images, calories in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            calories = calories.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(
                images
            ).squeeze(1)

            loss = criterion(
                outputs,
                calories
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            total_samples += (
                images.size(0)
            )

        torch.cuda.synchronize()
        dist.barrier()

        epoch_train_time = (
            time.time() - start
        )

        total_train_time += (
            epoch_train_time
        )

        # ----------------------------
        # VALIDATION
        # KHÔNG TÍNH VÀO TRAIN TIME
        # ----------------------------
        if rank == 0:

            val_mae, val_rmse = evaluate(
                model.module,
                val_loader,
                device
            )

            final_mae = val_mae
            final_rmse = val_rmse

            if val_mae < best_mae:

                best_mae = val_mae

                torch.save(
                    model.module.state_dict(),
                    "/kaggle/working/"
                    "resnet50_20epoch_2gpu_ddp_best.pth"
                )

            history.append({
                "epoch": epoch + 1,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "train_time_sec": epoch_train_time
            })

            print(
                f"Epoch {epoch+1:02d}/{EPOCHS} | "
                f"MAE: {val_mae:.2f} kcal | "
                f"RMSE: {val_rmse:.2f} kcal | "
                f"Train: {epoch_train_time:.2f}s"
            )

        # rank 1 đợi validation xong
        dist.barrier()


    # ========================================================
    # FINAL RESULT
    # ========================================================
    if rank == 0:

        throughput = (
            len(train_dataset)
            * EPOCHS
            / total_train_time
        )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            "/kaggle/working/"
            "resnet50_20epoch_2gpu_history.csv",
            index=False
        )

        print(
            "\n=== ResNet50 - 2 GPU DDP - 20 EPOCH RESULT ==="
        )

        print(
            f"Training time: "
            f"{total_train_time:.2f} s"
        )

        print(
            f"Throughput: "
            f"{throughput:.2f} images/s"
        )

        print(
            f"Best MAE: "
            f"{best_mae:.2f} kcal"
        )

        print(
            f"Final MAE: "
            f"{final_mae:.2f} kcal"
        )

        print(
            f"Final RMSE: "
            f"{final_rmse:.2f} kcal"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
