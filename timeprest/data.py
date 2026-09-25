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
    elif name == "tinyimagenet":
        train, test = tinyimagenet(data_cfg)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    train = _subset(train, data_cfg.get("train_subset"), seed)
    test = _subset(test, data_cfg.get("test_subset"), seed + 1)
    return train, test


class ArrayImages(Dataset):
    """uint8 images (N, 3, H, W) + labels held in memory; optional random crop (zero padding) and
    horizontal flip, then normalisation. Same augmentation as the CIFAR pipeline."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, mean, std, augment: bool, pad: int = 4):
        self.x, self.targets = x, y.long()
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.augment, self.pad = augment, pad

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        img = self.x[i].float().div_(255.0)
        if self.augment:
            h, w = img.shape[1:]
            p = self.pad
            img = torch.nn.functional.pad(img, (p, p, p, p))
            top, left = torch.randint(0, 2 * p + 1, (2,)).tolist()
            img = img[:, top:top + h, left:left + w]
            if torch.rand(()) < 0.5:
                img = img.flip(2)
        return (img - self.mean) / self.std, int(self.targets[i])


def _find_dir_with(root: str, marker: str) -> str:
    """First directory under `root` (following symlinks) that contains the file `marker`."""
    top = root.split("*")[0].rstrip("/\\") or "/"
    if os.path.exists(os.path.join(top, marker)):
        return top
    for d, subdirs, files in os.walk(top, followlinks=True):
        subdirs.sort()
        if marker in files:
            return d
    raise FileNotFoundError(f"no {marker} under {top}; attach the Tiny-ImageNet-200 dataset (folder tiny-imagenet-200/)")


def _load_tinyimagenet_arrays(folder: str):
    """Decode the original Tiny-ImageNet-200 layout: train/<wnid>/images/*.JPEG and
    val/images/*.JPEG labelled by val/val_annotations.txt (the test/ split has no labels, so the
    labelled val split is used as the test set). Class index = position in sorted wnids.txt."""
    import numpy as np
    from PIL import Image
    wnids = sorted(l.strip() for l in open(os.path.join(folder, "wnids.txt")) if l.strip())
    index = {w: k for k, w in enumerate(wnids)}

    def load(paths):
        out = np.empty((len(paths), 64, 64, 3), dtype=np.uint8)
        for k, path in enumerate(paths):
            with Image.open(path) as im:
                im = im.convert("RGB")
                if im.size != (64, 64):
                    raise ValueError(f"{path}: expected 64x64, got {im.size}")
                out[k] = np.asarray(im)
        return out

    tr_paths, tr_y = [], []
    for w in wnids:
        d = os.path.join(folder, "train", w, "images")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpeg", ".jpg", ".png")):
                tr_paths.append(os.path.join(d, f))
                tr_y.append(index[w])
    va_paths, va_y = [], []
    for line in open(os.path.join(folder, "val", "val_annotations.txt")):
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            va_paths.append(os.path.join(folder, "val", "images", parts[0]))
            va_y.append(index[parts[1]])
    return load(tr_paths), np.asarray(tr_y, dtype=np.int64), load(va_paths), np.asarray(va_y, dtype=np.int64)


def _channel_stats(x, chunk: int = 5000) -> tuple[list[float], list[float]]:
    """Per-channel mean/std of uint8 images (N, H, W, 3) scaled to [0, 1], accumulated in chunks
    (a float64 copy of all 100k Tiny-ImageNet images would need ~9 GiB)."""
    import numpy as np
    s = np.zeros(3)
    s2 = np.zeros(3)
    n = 0
    for i in range(0, len(x), chunk):
        c = x[i:i + chunk].reshape(-1, 3).astype(np.float64) / 255.0
        s += c.sum(0)
        s2 += (c * c).sum(0)
        n += c.shape[0]
    mean = s / n
    return mean.tolist(), np.sqrt(np.maximum(s2 / n - mean * mean, 0.0)).tolist()


def tinyimagenet(data_cfg: dict):
    """Tiny-ImageNet-200 decoded once into uint8 arrays cached as .npy (data.cache_dir), then held
    in memory. Normalisation statistics are computed on the training images."""
    import json

    import numpy as np
    folder = _find_dir_with(data_cfg["root"], "wnids.txt")
    cache = data_cfg.get("cache_dir") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "timeprest_tin")
    os.makedirs(cache, exist_ok=True)
    names = ["train_x", "train_y", "val_x", "val_y"]
    paths = {n: os.path.join(cache, n + ".npy") for n in names}
    stats_path = os.path.join(cache, "stats.json")
    ready = lambda: all(os.path.exists(p) for p in list(paths.values()) + [stats_path])
    if not ready():
        # one process decodes (lock file), the others wait for the finished cache
        lock = os.path.join(cache, ".building")
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            owner = True
        except FileExistsError:
            owner = False
        if not owner:
            import time
            t0 = time.time()
            while not ready():
                if time.time() - t0 > 3600:
                    raise TimeoutError(f"waited 1 h for {cache}; delete {lock} if a previous run crashed")
                time.sleep(2)
    if not ready():
        arrays = dict(zip(names, _load_tinyimagenet_arrays(folder)))
        mean, std = _channel_stats(arrays["train_x"])
        stats = {"mean": mean, "std": std, "source": folder,
                 "n_train": len(arrays["train_y"]), "n_val": len(arrays["val_y"])}
        for n, a in arrays.items():       # write-then-rename; stats.json last marks completion
            tmp = f"{paths[n]}.{os.getpid()}.tmp.npy"
            np.save(tmp, a)
            os.replace(tmp, paths[n])
        tmp = f"{stats_path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(stats, fh)
        os.replace(tmp, stats_path)          # written last: marks the cache as complete
        os.remove(lock)
    with open(stats_path) as fh:
        stats = json.load(fh)
    arr = {n: np.load(paths[n]) for n in names}
    to_chw = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()
    augment = data_cfg.get("augment", True)
    train = ArrayImages(to_chw(arr["train_x"]), torch.from_numpy(arr["train_y"]), stats["mean"], stats["std"], augment)
    test = ArrayImages(to_chw(arr["val_x"]), torch.from_numpy(arr["val_y"]), stats["mean"], stats["std"], False)
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
