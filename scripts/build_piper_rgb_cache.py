"""Sequentially decode each original train/val video once into a lossless RGB cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def decode_one(job):
    import numpy as np
    import torch
    import torchvision
    from openpi_client.image_tools import resize_with_pad
    torch.set_num_threads(1)
    source, output = Path(job["source"]), Path(job["output"])
    if sha256(source) != job["source_sha256"]:
        raise ValueError(f"Source video differs from immutable split: {source}")
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(source), "video")
    rgb_path, pts_path = output / (job["stem"] + ".npy"), output / (job["stem"] + "_pts.npy")
    if rgb_path.exists() or pts_path.exists():
        raise FileExistsError(rgb_path)
    rgb = np.lib.format.open_memmap(str(rgb_path) + ".tmp", mode="w+", dtype=np.uint8,
                                 shape=(job["length"], 224, 224, 3))
    timestamps = []
    for index, frame in enumerate(reader):
        if index >= job["length"]:
            raise ValueError(f"More video frames than episode rows: {source}")
        original = np.moveaxis(frame["data"].numpy(), 0, -1)
        rgb[index] = resize_with_pad(original, 224, 224)
        timestamps.append(frame["pts"])
    reader.container.close()
    if len(timestamps) != job["length"]:
        raise ValueError(f"Missing video frames: {source}")
    pts = np.asarray(timestamps, dtype=np.float32)
    if not np.all(np.diff(pts) > 0):
        raise ValueError("Video PTS must be strictly increasing")
    rgb.flush()
    del rgb
    Path(str(rgb_path) + ".tmp").rename(rgb_path)
    np.save(pts_path, pts)
    stat = source.stat()
    result = {"rgb":rgb_path.name, "pts":pts_path.name, "frames":len(pts),
              "source":str(source), "source_sha256":job["source_sha256"],
              "source_stat":[stat.st_size, stat.st_mtime_ns]}
    for name, path in (("rgb", rgb_path), ("pts", pts_path)):
        stat = path.stat()
        result[name + "_stat"] = [stat.st_size, stat.st_mtime_ns]
        result[name + "_sha256"] = sha256(path)
    return job["key"], result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    root = Path.cwd()
    split_path = root / "assets/pi05_piper_stage1/eggplant_potato/split.json"
    split = json.loads(split_path.read_text())
    episodes = set(split["splits"]["train"] + split["splits"]["val"])
    if episodes & set(split["splits"]["test"]):
        raise ValueError("Test split must remain untouched")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for episode in split["episodes"]:
        ep = episode["episode_index"]
        if ep not in episodes:
            continue
        for video in episode["videos"]:
            camera = Path(video["path"]).parent.name
            jobs.append({"key":f"{ep}:{camera}", "stem":f"ep{ep:06d}_{camera.split('.')[-1]}",
                         "source":str(root / "Datasets/eggplant_potato_gripper_binary" / video["path"]),
                         "source_sha256":video["sha256"], "length":episode["length"], "output":str(args.output)})
    manifest = {"schema":"piper_rgb224_cache_v1", "shape":[224,224,3], "dtype":"uint8",
                "split_sha256":sha256(split_path), "splits":["train","val"],
                "resize_source_sha256":sha256(root / "packages/openpi-client/src/openpi_client/image_tools.py"),
                "decoder":"same installed torchvision VideoReader/PyAV as original loader",
                "videos":{}, "started":started}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(decode_one, job) for job in jobs]
        for future in as_completed(futures):
            key, result = future.result()
            manifest["videos"][key] = result
            if len(manifest["videos"]) % 30 == 0:
                print(json.dumps({"completed":len(manifest["videos"]), "total":len(jobs), "seconds":time.time()-started}), flush=True)
    manifest["completed"] = time.time()
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.rename(args.output / "manifest.json")
    print(json.dumps({"complete":True, "videos":len(jobs), "seconds":time.time()-started}), flush=True)


if __name__ == "__main__":
    main()
