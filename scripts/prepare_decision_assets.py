"""Freeze reviewed sparse points and existing episode decisions without rewriting data."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path("assets/pi05_piper_decision_v1")
    review=Path("logs/rtc_grounding_pair_20260910/grounding_review")
    split_path=Path("assets/pi05_piper_reach_arm_v1/eggplant_potato/split.json")
    roles_path=Path("logs/arm_role_audit_20260910/dataset_arm_roles_v2.json")
    annotation_path=Path("Datasets/eggplant_potato_reach_arm_v1/meta/reach_arm_annotations.jsonl")
    splits=json.loads(split_path.read_text())["splits"]
    roles=json.loads(roles_path.read_text())["episodes"]
    annotations=[json.loads(s) for s in annotation_path.read_text().splitlines()]
    points=json.loads((review/"reviewed_points.json").read_text())
    decisions=[]
    for r in roles:
        ep=r["episode"];segments=[a for a in annotations if a["episode_index"]==ep]
        lid=next(a for a in segments if a["original_subtask"]=="reach the handle of the lid")
        obj=next(a for a in segments if a is not lid)
        split=next(s for s,episodes in splits.items() if ep in episodes)
        decisions.append(dict(episode=ep,split=split,prompt=r["task"],layout=r["layout_visual"],
            lid_arm=r["lid_arm_resolved"],lid_label=lid["subtask"],object_label=obj["subtask"],
            object_start=obj["start_frame"],object_end=obj["end_frame_exclusive"]))
    assert len(decisions)==198 and len(splits["train"])==178 and len(splits["val"])==20
    cache=Path(".stage1_staging/piper_rgb224_reach_arm_v1")
    for row in points["rows"]:
        assert row["reviewed"] and row["episode"] in splits[row["split"]]
        path=review/f"key_{row['id']:03d}.png"
        assert sha(path)==row["image_sha256"]
        actual=np.load(cache/f"ep{row['episode']:06d}_cam_high.npy",mmap_mode="r")[row["frame"]]
        np.testing.assert_array_equal(np.array(Image.open(path)),actual)
        for p in row["points"].values():
            if p["visible"]:
                xy=np.array(p["xy"])*224;box=p["bbox"]
                assert np.isfinite(xy).all() and box[0]<=xy[0]<box[2] and box[1]<=xy[1]<box[3]
    root.mkdir(exist_ok=False)
    (root/"decisions.json").write_text(json.dumps(decisions,indent=2)+"\n")
    (root/"reviewed_points.json").write_text(json.dumps(points,indent=2)+"\n")
    ready=dict(status="reviewed_and_verified",reviewer=points["reviewer"],
        files={n:sha(root/n) for n in ["decisions.json","reviewed_points.json"]},
        sources={str(p):sha(p) for p in [split_path,roles_path,annotation_path,cache/"manifest.json",Path(__file__)]},
        counts={s:dict(episodes=len({r["episode"] for r in points["rows"] if r["split"]==s}),
            frames=sum(r["split"]==s for r in points["rows"]),
            visible_points=sum(p["visible"] for r in points["rows"] if r["split"]==s for p in r["points"].values())) for s in splits},
        unchanged_dataset=True,exact_cached_pixels_verified=80)
    (root/"READY.json").write_text(json.dumps(ready,indent=2)+"\n")
    print(json.dumps(ready))


if __name__=="__main__":main()
