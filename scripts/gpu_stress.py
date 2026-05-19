"""CUDA matrix-multiply stress program.

Runs one process per selected GPU and keeps each device busy with repeated GEMM
work. Stop with Ctrl-C, SIGTERM, or `tmux kill-session -t gpu_stress`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import time

import torch


STOP = False


def request_stop(signum: int, frame: object) -> None:
    del signum, frame
    global STOP
    STOP = True


def selected_gpus(value: str) -> list[int]:
    if value.lower() == "all":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        return list(range(torch.cuda.device_count()))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def torch_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtypes[name]


def worker(gpu_id: int, size: int, dtype_name: str, batch: int) -> None:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    torch.cuda.set_device(gpu_id)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dtype = torch_dtype(dtype_name)
    left = torch.randn((size, size), device=gpu_id, dtype=dtype)
    right = torch.randn((size, size), device=gpu_id, dtype=dtype)
    out = torch.empty((size, size), device=gpu_id, dtype=dtype)
    torch.cuda.synchronize(gpu_id)

    print(
        f"[gpu {gpu_id}] pid={os.getpid()} size={size} dtype={dtype_name} batch={batch}",
        flush=True,
    )

    iterations = 0
    last_log = time.monotonic()
    while not STOP:
        for _ in range(batch):
            torch.mm(left, right, out=out)
        iterations += batch

        now = time.monotonic()
        if now - last_log >= 30:
            torch.cuda.synchronize(gpu_id)
            print(f"[gpu {gpu_id}] iterations={iterations}", flush=True)
            last_log = now

    torch.cuda.synchronize(gpu_id)
    print(f"[gpu {gpu_id}] stopped iterations={iterations}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Occupy CUDA GPUs with GEMM load.")
    parser.add_argument("--gpus", default="all", help="GPU ids like 0,1,2 or all.")
    parser.add_argument("--size", type=int, default=16384, help="Square GEMM size.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--batch", type=int, default=4, help="GEMMs queued before checking stop.")
    args = parser.parse_args()

    gpus = selected_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected.")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    mp.set_start_method("spawn", force=True)

    processes = [
        mp.Process(target=worker, args=(gpu_id, args.size, args.dtype, args.batch))
        for gpu_id in gpus
    ]
    for process in processes:
        process.start()

    try:
        while not STOP and any(process.is_alive() for process in processes):
            time.sleep(1)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)


if __name__ == "__main__":
    main()
