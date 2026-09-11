"""Create immutable R1 weak-boundary assets and an honest D0 review backlog."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.training.stage1_data import manifest_digest, sha256_file
from openpi.training.subtask_transition import evaluation_report

R0_SHA = "64600bed0b60e0529ac2721edb55717ac1c80fba2a607292857668a35e7a9275"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("Datasets/eggplant_potato_gripper_binary")
    old_assets = Path("assets/pi05_piper_stage1/eggplant_potato")
    manifest = json.loads((old_assets / "split.json").read_text())
    assert manifest_digest(manifest) == manifest["manifest_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    all_frames, all_events = {}, []
    for split in ["train", "val"]:
        frames = []
        for record in sorted(manifest["episodes"], key=lambda r:r["episode_index"]):
            if record["episode_index"] not in manifest["splits"][split]:
                continue
            path = root / record["parquet_path"]
            assert sha256_file(path) == record["parquet_sha256"]
            table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", "task", "subtask"]).to_pydict()
            labels, times = table["subtask"], table["timestamp"]
            assert table["frame_index"] == list(range(record["length"]))
            assert all(t2 > t1 for t1,t2 in zip(times,times[1:]))
            positions = [i for i in range(1,len(labels)) if labels[i] != labels[i-1]]
            events = []
            for k, pos in enumerate(positions):
                event = {"event_id":f"{split}:{record['episode_index']}:{pos}", "split":split,
                         "episode":record["episode_index"], "task":record["task"], "frame":pos,
                         "t_label":times[pos], "old":labels[pos-1], "new":labels[pos],
                         "transition":labels[pos-1]+" -> "+labels[pos],
                         "cell_start":(times[positions[k-1]]+times[pos])/2 if k else times[0],
                         "cell_end":(times[positions[k+1]]+times[pos])/2 if k+1<len(positions) else times[-1]+1/30,
                         "new_end_time":times[positions[k+1]] if k+1<len(positions) else times[-1]+1/30,
                         "videos":record["videos"], "review_status":"unreviewed",
                         "ready_lower":None, "ready_upper":None, "t_latest":None,
                         "supervision_kind":"original_annotation_only"}
                events.append(event)
            all_events.extend(events)
            for i in range(len(labels)):
                nearest = min(events,key=lambda e:abs(e["frame"]-i)) if events else None
                near = nearest if nearest and abs(nearest["frame"]-i)<=15 else None
                frames.append({"index":len(frames), "episode":record["episode_index"], "frame":i,
                               "timestamp":times[i], "episode_length":len(labels), "task":table["task"][i],
                               "label":labels[i], "boundary_event":near["event_id"] if near else None,
                               "transition":near["transition"] if near else None,
                               "boundary_side":("pre" if i<near["frame"] else "post") if near else None})
        all_frames[split]=frames
        (args.output/f"frames_{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in frames))
    (args.output/"boundary_index.jsonl").write_text("".join(json.dumps(r)+"\n" for r in all_events))
    (args.output/"handoff_annotations.jsonl").write_text("".join(json.dumps({k:e[k] for k in ["event_id","split","episode","review_status","ready_lower","ready_upper","t_latest","supervision_kind"]})+"\n" for e in all_events))
    prior=Path("logs/pi05_subtask_stage1/m3_native_dense_val")
    report=json.loads((prior/"report.json").read_text())
    assert report["weights_sha256"]==R0_SHA and report["split_sha256"]==manifest["manifest_sha256"]
    predictions={}
    for path in sorted(prior.glob("semantics_rank_*.json")):
        for row in json.loads(path.read_text()):
            key=(row["episode"],row["frame"])
            assert key not in predictions
            predictions[key]=row
    merged=[]
    for row in all_frames["val"]:
        saved=predictions.pop((row["episode"],row["frame"]))
        assert saved["label"]==row["label"] and saved["task"]==row["task"]
        merged.append(dict(row,prediction=saved["prediction"],status=saved["status"]))
    assert not predictions
    baseline=evaluation_report(merged,[e for e in all_events if e["split"]=="val"])
    baseline.update(checkpoint=report["checkpoint"],weights_sha256=R0_SHA, source="archived complete R0 dense validation")
    (args.output/"baseline_report.json").write_text(json.dumps(baseline,indent=2)+"\n")
    selected=[]
    for task in sorted({r["task"] for r in all_frames["train"]}):
        eps=sorted({r["episode"] for r in all_frames["train"] if r["task"]==task})
        selected.extend(np.asarray(eps)[np.linspace(0,len(eps)-1,10,dtype=int)].tolist())
    review=[e for e in all_events if e["split"]=="val" or e["episode"] in selected]
    material=["<!doctype html><meta charset='utf-8'><title>D0 review backlog</title>",
              "<h1>D0: original annotations; physical readiness unreviewed</h1>",
              "<p>Three views, +/-1 second. Do not infer grasp success from gripper command alone.</p>"]
    for e in review:
        material.append("<h2>"+html.escape(e["event_id"]+" | "+e["transition"])+"</h2>")
        for video in e["videos"]:
            material.append(f'<video controls preload="none" width="400" src="{(root/video["path"]).resolve()}#t={max(0,e["t_label"]-1):.3f},{e["t_label"]+1:.3f}"></video>')
    (args.output/"review.html").write_text("\n".join(material))
    (args.output/"review_queue.json").write_text(json.dumps(review,indent=2)+"\n")
    protocol={"schema_version":1,"variant":"r1_boundary_ce_v1","seed":42,"window_frames":15,
              "split_sha256":manifest["manifest_sha256"],"norm_sha256":sha256_file(old_assets/"norm_stats.json"),
              "parent_weights_sha256":R0_SHA,"label_shift_frames":0,"physical_readiness_reviewed":False,
              "scope":"Conservative boundary-CE refinement; readiness-based relabeling and deployment gate remain pending human review",
              "train_frames":len(all_frames["train"]),"val_frames":len(all_frames["val"]),
              "train_events":sum(e["split"]=="train" for e in all_events),"val_events":sum(e["split"]=="val" for e in all_events),
              "evaluation":{"period_seconds":0.764,"phases_seconds":[0,0.255,0.509],"hold_seconds":0.3,
                            "selection":"Proxy safeguards then lexicographic failed/late/jitter; full val; not physical certification",
                            "stable_em_drop_max":0.01,"annotation_early_increase_max_fraction":0.02},
              "files":{p.name:sha256_file(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print(json.dumps({k:v for k,v in protocol.items() if k not in ["files","evaluation"]}))


if __name__=="__main__":
    main()
