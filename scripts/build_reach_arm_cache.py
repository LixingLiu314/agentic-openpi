"""Verify and reuse unchanged RGB content; decode newly included episodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
from pathlib import Path
import time

from build_piper_rgb_cache import decode_one, sha256
from openpi.training.reach_arm_data import ASSETS_ROOT, DATASET_ROOT, SPLIT_FILE_SHA256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reuse",type=Path,default=Path(".stage1_staging/piper_rgb224_cache_v2"))
    p.add_argument("--workers",type=int,default=8)
    args=p.parse_args()
    args.output=args.output.absolute()
    args.reuse=args.reuse.absolute()
    args.output.mkdir(parents=True,exist_ok=False)
    begun=time.time()
    split_path=ASSETS_ROOT/"split.json"
    assert sha256(split_path)==SPLIT_FILE_SHA256
    split=json.loads(split_path.read_text())
    assert set(split["splits"])=={"train","val"}
    episodes=set(split["splits"]["train"]+split["splits"]["val"])
    old=json.loads((args.reuse/"manifest.json").read_text())
    resize_hash=sha256("packages/openpi-client/src/openpi_client/image_tools.py")
    assert old["schema"]=="piper_rgb224_cache_v1" and old["shape"]==[224,224,3]
    assert old["resize_source_sha256"]==resize_hash
    manifest=dict(schema="piper_rgb224_cache_v1",shape=[224,224,3],dtype="uint8",
                  split_sha256=SPLIT_FILE_SHA256,splits=["train","val"],resize_source_sha256=resize_hash,
                  decoder=old["decoder"],videos={},started=begun,
                  reused_from=str(args.reuse),reused_manifest_sha256=sha256(args.reuse/"manifest.json"))
    jobs=[]
    reused=0
    for ep in split["episodes"]:
        assert ep["episode_index"] in episodes
        for video in ep["videos"]:
            camera=Path(video["path"]).parent.name
            key=f'{ep["episode_index"]}:{camera}'
            source=(DATASET_ROOT/video["path"]).absolute()
            assert sha256(source)==video["sha256"]
            if key in old["videos"]:
                record=copy.deepcopy(old["videos"][key])
                assert record["source_sha256"]==video["sha256"] and record["frames"]==ep["length"]
                for field in ("rgb","pts"):
                    path=args.reuse/record[field]
                    assert sha256(path)==record[field+"_sha256"], str(path)
                    stat=path.stat()
                    assert [stat.st_size,stat.st_mtime_ns]==record[field+"_stat"]
                    (args.output/record[field]).symlink_to(path)
                stat=source.stat()
                record.update(source=str(source),source_stat=[stat.st_size,stat.st_mtime_ns])
                manifest["videos"][key]=record
                reused+=1
                if reused%120==0: print(json.dumps(dict(verified_reused_videos=reused)),flush=True)
            else:
                jobs.append(dict(key=key,stem=f'ep{ep["episode_index"]:06d}_{camera.split(".")[-1]}',
                                 source=str(source),source_sha256=video["sha256"],length=ep["length"],output=str(args.output)))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(decode_one,job) for job in jobs]
        for future in as_completed(futures):
            key,record=future.result(); manifest["videos"][key]=record
            if (len(manifest["videos"])-reused)%15==0:
                print(json.dumps(dict(decoded_new_videos=len(manifest["videos"])-reused,total_new=len(jobs))),flush=True)
    assert len(manifest["videos"])==594
    manifest.update(completed=time.time(),verified_reused_videos=reused,decoded_new_videos=len(jobs))
    temp=args.output/"manifest.json.tmp"
    temp.write_text(json.dumps(manifest,indent=2)+"\n");temp.rename(args.output/"manifest.json")
    print(json.dumps(dict(passed=True,reused=reused,new=len(jobs),videos=594,seconds=time.time()-begun)),flush=True)


if __name__=="__main__": main()
