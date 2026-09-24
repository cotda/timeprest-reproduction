import os

# Load every CUDA kernel up front. PyTorch/driver default is LAZY: a kernel's first launch loads its
# module, which needs a context-wide sync and so blocks behind a posted NCCL receive that waits for
# the peer -> deadlock (CUDA docs, "Lazy Loading: concurrent execution"). The driver reads this at
# CUDA initialisation (even torch.cuda.is_available() triggers it), so set it on package import.
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
