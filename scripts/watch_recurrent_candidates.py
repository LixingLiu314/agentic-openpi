"""Emit completed candidate events through one SSH stream, using inotify."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    library = ctypes.CDLL(None, use_errno=True)
    fd = library.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0 or library.inotify_add_watch(fd, os.fsencode(args.root), 0x8 | 0x80 | 0x100) < 0:
        raise OSError(ctypes.get_errno(), "Cannot watch experiment completion directory")
    delivered = set()
    deadline = time.monotonic() + args.hours * 3600
    try:
        while time.monotonic() < deadline:
            for arm in ("stateless", "recurrent"):
                path = args.root / (arm + "_complete.json")
                if path.is_file() and arm not in delivered:
                    event = json.loads(path.read_text())
                    assert event["arm"] == arm and event["native_gate_passed"] is True
                    print(json.dumps(dict(event="candidate", **event)), flush=True)
                    delivered.add(arm)
            failure = args.root / "formal_failure.json"
            if failure.exists():
                print(json.dumps(dict(event="failure", detail=json.loads(failure.read_text()))), flush=True)
                return
            if len(delivered) == 2:
                print(json.dumps(dict(event="complete")), flush=True)
                return
            ready, _, _ = select.select([fd], [], [], min(60, max(0, deadline - time.monotonic())))
            if ready:
                os.read(fd, 65536)
        raise TimeoutError("Candidate watcher deadline elapsed")
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
