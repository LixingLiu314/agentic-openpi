"""Build a causal approximate replay from the user's failed trial, without overlays."""
import argparse
import hashlib
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output
    if (root/'protocol.json').exists():raise FileExistsError('Preserve the fixed stress protocol')
    run=json.loads((root/'source/run.json').read_text());frames=run['video']['frames']
    cameras=['cam_high','cam_left_wrist','cam_right_wrist'];records=[];missing=[]
    for query in run['queries']:
        chosen={}
        for view in cameras:
            target=query['observation']['sensor_ros_timestamps'][view]
            available=[f for f in frames if f['sensor_ros_timestamps'][view]<=target+1e-7]
            if not available:break
            frame=min(available,key=lambda f:(target-f['sensor_ros_timestamps'][view],abs(f['capture_monotonic']-query['observation']['snapshot_monotonic'])))
            chosen[view]={'frame':frame['frame'],'camera_lag_seconds':target-frame['sensor_ros_timestamps'][view],
                          'same_camera_timestamp':abs(target-frame['sensor_ros_timestamps'][view])<1e-7}
        if len(chosen)!=3:missing.append(query['query']);continue
        records.append({'query':query['query'],'timestamp':query['observation']['snapshot_monotonic']-run['video']['start_monotonic'],
                        'prompt':run['config']['prompt'],'state':query['state'],'camera_selection':chosen,'file':f"query_{query['query']:04d}.npz"})
    needed={r['camera_selection'][v]['frame'] for r in records for v in cameras};decoded={}
    with av.open(str(root/'source/video.mp4')) as video:
        for i,frame in enumerate(video.decode(video=0)):
            if i in needed:
                rgb=frame.to_ndarray(format='rgb24')
                if rgb.shape!=(576,1440,3):raise ValueError('Recording layout differs from audited renderer')
                decoded[i]=rgb[92:452].copy()
    for r in records:
        data={'state':np.asarray(r['state'],dtype=np.float32)}
        for col,view in enumerate(cameras):data[view]=decoded[r['camera_selection'][view]['frame']][:,col*480:(col+1)*480].copy()
        np.savez_compressed(root/r['file'],**data);r['sha256']=digest(root/r['file'])
    protocol={'schema_version':1,'run':'20260908_105241_afe075d0b3','observations':len(records),'missing_queries':missing,
              'source_hashes':{name:digest(root/'source'/name) for name in ['run.json','video.mp4']},
              'input_scope':'Lossy 480x360 camera crops; source camera timestamp never newer than model input. All prediction overlays removed. Approximate replay, not original RGB.',
              'constraints':{'forbidden_final_put_queries':[r['query'] for r in records],
                             'closed_gripper_queries':[94,95],
                             'closed_gripper_allowed_text':['Grasp the handle of the lid','Move away the lid'],
                             'required_closed_gripper_allowed_count':1,'max_forbidden_final_put':0,'max_invalid_generations':1},
              'review_basis':'Assistant visual review of complete sampled trial: eggplant remains outside closed box; no placement sequence. q94/q95 state shows closure near lid handle. This supports phase constraints, not reliable grasp/physical readiness certification.',
              'split_use':'Post-selection diagnostic only; never fit or calibration. Preserve failed diagnostic results.',
              'physical_readiness_reviewed':False,'records':records}
    if any(q not in {r['query'] for r in records} for q in [94,95]):raise ValueError('Critical observations not reconstructable')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    chosen=[r for r in records if r['query'] in [1,70,94,95,99]]
    canvas=Image.new('RGB',(1440,len(chosen)*390),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for i,r in enumerate(chosen):
        data=np.load(root/r['file']);draw.text((8,i*390+3),f"Approximate input | query {r['query']} | t={r['timestamp']:.3f}s",font=font,fill='black')
        for col,view in enumerate(cameras):canvas.paste(Image.fromarray(data[view]),(col*480,i*390+30))
    canvas.save(root/'replay_inputs.jpg',quality=92)
    print(json.dumps({'observations':len(records),'missing':missing,'critical_queries_present':True,'protocol':str(root/'protocol.json')}))


if __name__=='__main__':main()
