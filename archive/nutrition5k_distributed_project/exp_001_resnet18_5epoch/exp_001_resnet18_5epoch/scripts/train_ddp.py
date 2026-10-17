
import os
import time
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
from torchvision.models import resnet18, ResNet18_Weights


class NutritionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        calories = torch.tensor(row["calories"], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, calories


transform = transforms.Compose([
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


def evaluate(model, loader, criterion, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for images, calories in loader:
            images = images.to(device)
            calories = calories.to(device)

            outputs = model(images).squeeze(1)

            preds.extend(outputs.cpu().numpy())
            targets.extend(calories.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    return mae, rmse


def main():

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    train_df = pd.read_csv("/kaggle/working/train.csv")
    val_df = pd.read_csv("/kaggle/working/val.csv")

    train_dataset = NutritionDataset(train_df, transform)
    val_dataset = NutritionDataset(val_df, val_transform)

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    EPOCHS = 5

    dist.barrier()
    start = time.time()

    for epoch in range(EPOCHS):

        sampler.set_epoch(epoch)
        model.train()

        total_loss = 0

        for images, calories in train_loader:

            images = images.to(device)
            calories = calories.to(device)

            optimizer.zero_grad()

            outputs = model(images).squeeze(1)

            loss = criterion(outputs, calories)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        dist.barrier()

        if rank == 0:
            mae, rmse = evaluate(
                model.module,
                val_loader,
                criterion,
                device
            )

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"MAE: {mae:.2f} kcal | "
                f"RMSE: {rmse:.2f} kcal"
            )

        dist.barrier()

    total_time = time.time() - start

    if rank == 0:

        throughput = (
            len(train_dataset) * EPOCHS
        ) / total_time

        print("\n=== DDP RESULT ===")
        print(f"GPU count: {world_size}")
        print(f"Training time: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} images/sec")
        print(f"Final MAE: {mae:.2f} kcal")
        print(f"Final RMSE: {rmse:.2f} kcal")

        torch.save(
            model.module.state_dict(),
            "/kaggle/working/resnet18_2gpu_ddp.pth"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
