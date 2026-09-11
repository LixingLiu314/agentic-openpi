"""Propose sparse target points on exact RGB224 frames; review is mandatory.

Color/shape heuristics are annotation aids for this dataset only, never model
inputs or deployment rules. Original parquet/video/cache assets are read-only.
"""
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def proposals(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    h,s,v=hsv.transpose(2,0,1);r,g,b=rgb.astype(np.float32).transpose(2,0,1)
    masks={"sweet potato":(r>1.4*g)&(b>1.3*g)&(r>.85*b)&(r>55)&(s>55),
           "eggplant":(v<100)}
    result={}
    for name,mask in masks.items():
        mask[:58]=False;mask[165:]=False;mask[:,:6]=False;mask[:,218:]=False
        n,ids,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        candidates=[]
        for i in range(1,n):
            x,y,w,hh,area=map(int,stats[i]);cx,cy=centers[i]
            if not 8<=area<=220 or max(w,hh)<7 or min(w,hh)<2:continue
            ys,xs=np.nonzero(ids==i)
            _,(rw,rh),_=cv2.minAreaRect(np.stack([xs,ys],axis=1).astype(np.float32))
            elongation=max(rw,rh)/max(1,min(rw,rh))
            if name=="eggplant" and (elongation<1.8 or cy>132 or not 10<cx<210):continue
            pixels=rgb[ids==i].astype(float)
            score=area*(1 if cy<148 else .45)
            if name=="eggplant":score*=min(elongation,4)
            candidates.append(dict(xy=[float(cx/224),float(cy/224)],bbox=[x,y,x+w,y+hh],
                                   area=area,score=float(score),mean_rgb=pixels.mean(0).tolist()))
        candidates.sort(key=lambda z:-z["score"])
        result[name]=dict(selected=candidates[0] if candidates else None,candidates=candidates)
    return result


def main():
    out=Path("logs/rtc_grounding_pair_20260910/grounding_review");out.mkdir(parents=True,exist_ok=True)
    roles=json.loads(Path("logs/arm_role_audit_20260910/dataset_arm_roles_v2.json").read_text())["episodes"]
    splits=json.loads(Path("assets/pi05_piper_reach_arm_v1/eggplant_potato/split.json").read_text())["splits"]
    annotations=[json.loads(s) for s in Path("Datasets/eggplant_potato_reach_arm_v1/meta/reach_arm_annotations.jsonl").read_text().splitlines()]
    reaches={(x["episode_index"],x["original_subtask"]):x for x in annotations}
    lookup={r["episode"]:r for r in roles}
    groups=defaultdict(list)
    for split,episodes in splits.items():
        for ep in episodes:
            r=lookup[ep];groups[(split,r["task"],r["layout_visual"],r["lid_arm_resolved"])].append(ep)
    rng=np.random.default_rng(42);selected=[]
    for key,episodes in sorted(groups.items()):
        for ep in rng.choice(sorted(episodes),size=min(4 if key[0]=="train" else 1,len(episodes)),replace=False):
            selected.append((key[0],int(ep)))
    rows=[];cache=Path(".stage1_staging/piper_rgb224_reach_arm_v1")
    for split,ep in selected:
        role=lookup[ep];target="sweet potato" if "sweet potato" in role["task"] else "eggplant"
        for stage,frame in [("start",0),("reach_object",reaches[(ep,"reach the "+target)]["start_frame"])]:
            rgb=np.load(cache/f"ep{ep:06d}_cam_high.npy",mmap_mode="r")[frame].copy()
            label=reaches[(ep,"reach the handle of the lid" if stage=="start" else "reach the "+target)]["subtask"]
            row=dict(id=len(rows),split=split,episode=ep,frame=frame,stage=stage,target=target,prompt=role["task"],
                     subtask=label,layout=role["layout_visual"],lid_arm=role["lid_arm_resolved"],
                     camera="cam_high",image_size=[224,224],proposals=proposals(rgb),reviewed=False)
            Image.fromarray(rgb).save(out/f"key_{row['id']:03d}.png")
            rows.append(row)
    (out/"proposals.json").write_text(json.dumps(rows,indent=2))
    for start in range(0,len(rows),20):
        sheet=Image.new("RGB",(1120,1032),"white");d=ImageDraw.Draw(sheet)
        for offset,row in enumerate(rows[start:start+20]):
            x,y=224*(offset%5),258*(offset//5)
            image=Image.open(out/f"key_{row['id']:03d}.png").convert("RGB");draw=ImageDraw.Draw(image)
            for name,color in [("eggplant","cyan"),("sweet potato","yellow")]:
                p=row["proposals"][name]["selected"]
                if p:
                    draw.rectangle(p["bbox"],outline=color,width=1)
                    xx,yy=np.asarray(p["xy"])*224;draw.line((xx-3,yy,xx+3,yy),fill=color);draw.line((xx,yy-3,xx,yy+3),fill=color)
            sheet.paste(image,(x,y+34));d.text((x+3,y+2),f"id{row['id']} ep{row['episode']} f{row['frame']} {row['split']}",fill="black")
            d.text((x+3,y+16),f"goal {row['target']} / {row['stage']}",fill="black")
        sheet.save(out/f"review_{start//20:02d}.jpg",quality=95)
    print(json.dumps(dict(frames=len(rows),episodes=len(selected),strata=len(groups),
         missing_proposals=sum(p["selected"] is None for row in rows for p in row["proposals"].values()),
         status="proposed only; not approved training labels")))


if __name__=="__main__":main()
