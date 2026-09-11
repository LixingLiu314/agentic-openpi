"""Resume official audit downloads with generation-pinned checked HTTP ranges."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.parse
import urllib.request

import audit_pi05_base_weights as audit

ORIGINAL_DOWNLOAD = audit.download_object
BLOCK_BYTES = 8 * 1024**2


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download_ranges(item, root):
    prefix = "checkpoints/pi05_base/"
    if not item["name"].startswith(prefix):
        raise ValueError("Unexpected official object")
    relative = Path(item["name"][len(prefix) :])
    target = root / relative
    if relative.is_absolute() or ".." in relative.parts or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Object escapes audit directory")
    if target.exists():
        return ORIGINAL_DOWNLOAD(item, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".ranges.partial")
    journal = target.with_name(target.name + ".ranges.json")
    old_partial = target.with_name(target.name + ".partial")
    identity = {
        "name": item["name"],
        "generation": item["generation"],
        "size": int(item["size"]),
        "block_bytes": BLOCK_BYTES,
    }
    if journal.exists():
        progress = json.loads(journal.read_text())
        if progress["identity"] != identity or not partial.exists():
            raise ValueError("Partial range download identity changed")
    else:
        if partial.exists():
            raise ValueError("Partial ranges have no journal")
        if old_partial.exists():
            shutil.copyfile(old_partial, partial)
        else:
            partial.touch(exist_ok=False)
        prefix_bytes = partial.stat().st_size
        if prefix_bytes > identity["size"]:
            raise ValueError("Original partial exceeds official object size")
        progress = {"identity": identity, "prefix_bytes": prefix_bytes, "completed_starts": []}
        atomic_json(journal, progress)
    url = (
        "https://storage.googleapis.com/openpi-assets/"
        + urllib.parse.quote(item["name"], safe="/")
        + "?generation="
        + item["generation"]
    )
    done = set(progress["completed_starts"])
    starts = [start for start in range(progress["prefix_bytes"], identity["size"], BLOCK_BYTES) if start not in done]
    lock = threading.Lock()
    descriptor = os.open(partial, os.O_RDWR)

    def fetch(start):
        end = min(start + BLOCK_BYTES, identity["size"]) - 1
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{identity['size']}"
                    ):
                        raise ValueError("Server did not honor the exact pinned byte range")
                    position = start
                    while block := response.read(min(1024**2, end + 1 - position)):
                        offset = 0
                        while offset < len(block):
                            offset += os.pwrite(descriptor, block[offset:], position + offset)
                        position += len(block)
                    if position != end + 1:
                        raise ValueError("Incomplete official byte range")
                os.fsync(descriptor)
                with lock:
                    done.add(start)
                    progress["completed_starts"] = sorted(done)
                    atomic_json(journal, progress)
                    if len(done) % 16 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "range_progress",
                                    "object": relative.name,
                                    "completed_blocks": len(done),
                                    "prefix_bytes": progress["prefix_bytes"],
                                }
                            ),
                            flush=True,
                        )
                return
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(fetch, starts))
    finally:
        os.close(descriptor)
    if partial.stat().st_size != identity["size"]:
        raise ValueError("Downloaded object length mismatch")
    if "md5Hash" in item:
        actual = base64.b64encode(audit.digest(partial, "md5").digest()).decode()
        expected = item["md5Hash"]
    else:
        import google_crc32c

        checksum = google_crc32c.Checksum()
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
                checksum.update(block)
        actual, expected = base64.b64encode(checksum.digest()).decode(), item["crc32c"]
    if actual != expected:
        raise ValueError(f"Downloaded object checksum mismatch: {relative}")
    partial.rename(target)
    return ORIGINAL_DOWNLOAD(item, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/pi05_piper_stage1/provenance/official_pi05_base_gcs_manifest.json"),
    )
    parser.add_argument("--official-root", type=Path, default=Path("checkpoints/pi05_base_official_jax_audit"))
    parser.add_argument("--local-weights", type=Path, default=Path("checkpoints/pi05_base_pytorch/model.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("logs/pi05_subtask_stage1/official_base_weight_audit.json"))
    args = parser.parse_args()
    provenance = args.output.with_name(args.output.stem + "_range_downloader.json")
    with provenance.open("x") as stream:
        json.dump(
            {
                "source": __file__,
                "source_sha256": audit.digest(Path(__file__), "sha256").hexdigest(),
                "original_audit_source_sha256": audit.digest(Path(audit.__file__), "sha256").hexdigest(),
                "objects_parallel": 4,
                "ranges_per_object": 6,
                "block_bytes": BLOCK_BYTES,
            },
            stream,
            indent=2,
        )
    audit.download_object = download_ranges
    audit.audit(args)
