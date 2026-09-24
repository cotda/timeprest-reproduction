"""Datasets and loaders. Data order depends only on (seed, epoch), so resume is reproducible."""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


def _synthetic(n: int, num_classes: int, shape, seed: int) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, *shape, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return TensorDataset(x, y)


def build_datasets(data_cfg: dict, num_classes: int, seed: int, input_shape=(3, 32, 32)):
    name = data_cfg["dataset"]
    if name == "synthetic":
        train = _synthetic(data_cfg.get("synthetic_train", 2048), num_classes, input_shape, seed)
        test = _synthetic(data_cfg.get("synthetic_test", 512), num_classes, input_shape, seed + 1)
    elif name in ("cifar100", "cifar10"):
        import torchvision
        import torchvision.transforms as T
        mean, std = STATS[name]
        norm = [T.ToTensor(), T.Normalize(mean, std)]
        train_tf = T.Compose(([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
                              if data_cfg.get("augment", True) else []) + norm)
        cls = torchvision.datasets.CIFAR100 if name == "cifar100" else torchvision.datasets.CIFAR10
        data_cfg = dict(data_cfg, root=resolve_root(data_cfg["root"], cls.base_folder))
        folder = os.path.join(data_cfg["root"], cls.base_folder)
        if not data_cfg.get("download", True) and not os.path.isdir(folder):
            raise FileNotFoundError(f"{folder} not found (data.download=false). Mount Drive or fix data.root; "
                                    f"the folder must contain the files of the original CIFAR python archive")
        train = cls(data_cfg["root"], train=True, download=data_cfg.get("download", True), transform=train_tf)
        test = cls(data_cfg["root"], train=False, download=data_cfg.get("download", True),
                   transform=T.Compose(norm))
    else:
        raise ValueError(f"unknown dataset {name!r}")
    train = _subset(train, data_cfg.get("train_subset"), seed)
    test = _subset(test, data_cfg.get("test_subset"), seed + 1)
    return train, test


def _subset(ds: Dataset, k, seed: int) -> Dataset:
    if not k or k >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:k].tolist()
    return Subset(ds, idx)


def train_loader(ds: Dataset, batch_size: int, seed: int, epoch: int, num_workers: int,
                 drop_last: bool = True, pin_memory: bool = False) -> DataLoader:
    g = torch.Generator().manual_seed(seed * 100003 + epoch)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=False)


def eval_loader(ds: Dataset, batch_size: int, num_workers: int, pin_memory: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory)


class RepeatBatch(Dataset):
    """One fixed batch repeated `repeats` times (overfit check)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, repeats: int):
        self.x, self.y, self.repeats = x, y, repeats

    def __len__(self):
        return self.repeats

    def __getitem__(self, idx):
        return self.x, self.y


def fixed_batch_loader(x: torch.Tensor, y: torch.Tensor, repeats: int) -> DataLoader:
    return DataLoader(RepeatBatch(x, y, repeats), batch_size=None, shuffle=False)


def resolve_root(root: str, base_folder: str, cache: str | None = None) -> str:
    """Return a directory that contains `base_folder` (e.g. cifar-100-python/).

    `root` may contain a glob (e.g. /kaggle/input/**): the tree is searched (following symlinks)
    for (1) a folder named `base_folder`, (2) any folder holding the CIFAR python files
    (train, test, meta / batches.meta) -- Kaggle drops the uploaded folder's own name -- or
    (3) the original archive `<base_folder>.tar.gz`. Cases 2-3 are exposed under `cache`
    (symlink / extraction) so torchvision finds `<cache>/<base_folder>`."""
    if os.path.isdir(os.path.join(root, base_folder)):
        return root
    if "*" not in root and not os.path.isdir(root):
        return root   # e.g. download target that does not exist yet
    import tarfile
    top = root.split("*")[0].rstrip("/\\") or "/"
    files = {"cifar-100-python": {"train", "test", "meta"},
             "cifar-10-batches-py": {"data_batch_1", "test_batch", "batches.meta"}}.get(base_folder, set())
    cache = cache or os.path.join(os.environ.get("TMPDIR", "/tmp"), "timeprest_data")
    named = plain = archive = None
    for d, subdirs, fnames in os.walk(top, followlinks=True):
        subdirs.sort()
        if os.path.basename(d) == base_folder and named is None:
            named = d
        if files and files <= set(fnames) and plain is None:
            plain = d
        if base_folder + ".tar.gz" in fnames and archive is None:
            archive = os.path.join(d, base_folder + ".tar.gz")
    if named:
        return os.path.dirname(named)
    target = os.path.join(cache, base_folder)
    os.makedirs(cache, exist_ok=True)
    if plain:
        if not os.path.exists(target):
            try:
                os.symlink(plain, target, target_is_directory=True)
            except FileExistsError:   # another rank was faster
                pass
            except OSError:           # no symlink permission (e.g. Windows): copy the files
                import shutil
                shutil.copytree(plain, target, dirs_exist_ok=True)
        return cache
    if archive:
        if not os.path.exists(target):
            tmp = os.path.join(cache, f".extract_{os.getpid()}")
            with tarfile.open(archive) as tf:
                tf.extractall(tmp)
            try:
                os.rename(os.path.join(tmp, base_folder), target)
            except OSError:           # another rank finished first
                pass
        return cache
    raise FileNotFoundError(f"no {base_folder}/ (nor its files train/test/meta, nor {base_folder}.tar.gz) "
                            f"under {top}; check that the Kaggle Dataset is attached (Add Input)")


def dataset_targets(ds) -> torch.Tensor:
    """All labels of a dataset without loading images (CIFAR, TensorDataset, Subset)."""
    if isinstance(ds, Subset):
        return dataset_targets(ds.dataset)[torch.as_tensor(ds.indices)]
    if isinstance(ds, TensorDataset):
        return ds.tensors[1].long()
    if hasattr(ds, "targets"):
        return torch.as_tensor(ds.targets).long()
    return torch.as_tensor([ds[i][1] for i in range(len(ds))]).long()


def epoch_order(n: int, seed: int, epoch: int) -> torch.Tensor:
    """Sample order of an epoch, identical on every rank."""
    g = torch.Generator().manual_seed(seed * 100003 + epoch)
    return torch.randperm(n, generator=g)
