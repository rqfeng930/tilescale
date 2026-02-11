"""
Minimal NVSHMEM host-side example.

This script:
- Initializes torch.distributed (NCCL backend).
- Initializes NVSHMEM via pynvshmem.init_nvshmem_by_uniqueid().
- Prints nvshmem_my_pe() and nvshmem_n_pes() on each rank.
- Calls nvshmem_barrier_all() once.

Usage (from repo root, after installing NVSHMEM from source + pynvshmem):

  # Environment:
  export NVSHMEM_SRC=/root/tilescale/3rdparty/nvshmem_src
  export LD_LIBRARY_PATH="$NVSHMEM_SRC/build/src/lib:$LD_LIBRARY_PATH"

  # Single node, 1 GPU:
  GPUS=1 PYTHON_EXEC=python3 ./tilelang/distributed/launch.sh \\
      examples/distributed/example_nvshmem_minimal.py

  # Single node, 2 GPUs:
  GPUS=2 PYTHON_EXEC=python3 ./tilelang/distributed/launch.sh \\
      examples/distributed/example_nvshmem_minimal.py
"""

import os

import torch
import torch.distributed as dist


def main() -> None:
    # Make sure we are in CUDA mode
    assert torch.cuda.is_available(), "CUDA is not available."

    import pynvshmem

    # Read ranks from env (launch.sh / torchrun sets these)
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    RANK = int(os.environ.get("RANK", "0"))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

    # Initialize torch.distributed (NCCL backend)
    dist.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=torch.distributed.timedelta(seconds=1800) if hasattr(torch.distributed, "timedelta") else None,  # type: ignore[arg-type]
    )
    torch.cuda.set_device(LOCAL_RANK)
    assert dist.is_initialized()

    TP_GROUP = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    # Initialize NVSHMEM using torch process group
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    my_pe = pynvshmem.nvshmem_my_pe()
    n_pes = pynvshmem.nvshmem_n_pes()

    print(
        f"[rank {RANK}/{WORLD_SIZE}, local_rank {LOCAL_RANK}] "
        f"nvshmem_my_pe={my_pe}, nvshmem_n_pes={n_pes}, "
        f"CUDA device={torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
    )

    # Simple global barrier across all PEs
    pynvshmem.nvshmem_barrier_all()
    print(f"[rank {RANK}] nvshmem_barrier_all() OK")


if __name__ == "__main__":
    # Helpful default in case user forgot
    os.environ.setdefault("TILELANG_USE_NVSHMEM", "1")
    main()

