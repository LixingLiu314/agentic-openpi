"""FIT-only observed grasp timing, inferred acting hand, and phase sidecar audit.

Acting hand is inferred for offline analysis from observed closing motion within
the annotated grasp span. Neither this rule nor any labels enter R2 inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq


def stats(values):
    values=np.asarray(values,dtype=float)
    return {'count':len(values),'p05_p50_p95':np.percentile(values,[5,50,95]).tolist()} if len(values) else {'count':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    protocol_path=Path('assets/pi05_piper_transition/eggplant_potato/r2_v1/protocol.json')
    protocol=json.loads(protocol_path.read_text());root=Path('Datasets/eggplant_potato_gripper_binary')
    items=[];hashes={};all_phase_frames={}
    for episode in protocol['fit_episodes']:
        path=root/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
        hashes[str(episode)]=hashlib.sha256(path.read_bytes()).hexdigest()
        table=pq.read_table(path,columns=['timestamp','subtask','observation.state']).to_pydict()
        stamps=np.array(table['timestamp']);labels=table['subtask'];states=np.array(table['observation.state'])
        edges=np.array([i for i in range(1,len(labels)) if labels[i]!=labels[i-1]])
        for start,end in zip([0,*edges.tolist()],[*edges.tolist(),len(labels)],strict=True):
            label=labels[start];all_phase_frames[label]=all_phase_frames.get(label,0)+end-start
            if 'grasp' not in label.lower():continue
            before=np.median(states[max(0,start-6):start][:,[6,13]],axis=0)
            after=np.median(states[max(start,end-6):end][:,[6,13]],axis=0)
            closing=before-after;hand=int(np.argmax(closing));column=[6,13][hand]
            resolved=float(closing[hand])>=.01 and float(closing[hand]-closing[1-hand])>=.01
            reference={}
            for threshold in [.02,.04,.06]:
                crossings=[i for i in range(max(1,start-75),min(len(labels)-2,end+1))
                           if states[i-1,column]>=threshold and np.all(states[i:i+3,column]<threshold)]
                index=min(crossings,key=lambda i:abs(stamps[i]-stamps[start])) if crossings else None
                reference[str(threshold)]={'frame':index,'offset_seconds':float(stamps[index]-stamps[start]) if index is not None else None}
            unmasked=[i for i in range(start,end) if min(abs(i-edges))>3]
            items.append({'episode':episode,'label':label,'start':int(start),'end_exclusive':int(end),'frames':int(end-start),
                          'duration_seconds':float(stamps[end-1]-stamps[start]+1/30),
                          'acting_hand':'left' if hand==0 else 'right','hand_resolved':resolved,
                          'closing_deltas':closing.tolist(),'terminal_active_width':float(after[hand]),
                          'D_unmasked_frames_if_active_known':len(unmasked),'closure_references':reference})
    labels=sorted({r['label'] for r in items});summary={}
    for label in labels:
        group=[r for r in items if r['label']==label];resolved=[r for r in group if r['hand_resolved']]
        summary[label]={'episodes':len(group),'hand_counts':{h:sum(r['acting_hand']==h for r in resolved) for h in ['left','right']},
                        'unresolved_hands':len(group)-len(resolved),'duration_seconds':stats([r['duration_seconds'] for r in group]),
                        'zero_unmasked_D_spans':sum(r['D_unmasked_frames_if_active_known']==0 for r in group),
                        'terminal_width':stats([r['terminal_active_width'] for r in resolved]),
                        'terminal_width_by_hand':{h:stats([r['terminal_active_width'] for r in resolved if r['acting_hand']==h]) for h in ['left','right']},
                        'threshold_offsets_seconds':{t:stats([r['closure_references'][t]['offset_seconds'] for r in resolved
                                 if r['closure_references'][t]['offset_seconds'] is not None]) for t in ['0.02','0.04','0.06']}}
    selected=[]
    for label in labels:
        # Median-duration examples for both role assignments; no validation scenes selected.
        for hand in ['left','right']:
            group=[r for r in items if r['label']==label and r['acting_hand']==hand and r['hand_resolved']]
            group.sort(key=lambda r:(r['duration_seconds'],r['episode']))
            selected.append(group[len(group)//2])
    report={'version':2,'scope':'138 FIT episodes only; no model or held-out data inference',
            'corrects_v1':'v1 closing-offset columns assumed lid=right/object=left; active hand actually varies. Use v2 for those offsets.',
            'summary':summary,'all_phase_frame_counts':all_phase_frames,'examples_for_visual_review':selected,'grasp_spans':items,
            'physical_readiness_reviewed':False,'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'parquet_sha256':hashes,
            'limitations':'Width thresholds and closure-derived role assignments describe motion only; neither certifies contact, successful grasp, or successor readiness.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,a.output/Path(__file__).name)
    print(json.dumps({'summary':summary,'visual_examples':[(r['episode'],r['label'],r['acting_hand']) for r in selected]},indent=2),flush=True)


if __name__=='__main__':main()
