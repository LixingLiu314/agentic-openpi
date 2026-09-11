"""Read-only FIT-only phase duration/gripper audit; thresholds are NOT readiness labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def summary(values):
    values=np.asarray(values,dtype=float)
    return {'count':len(values),'p10_p50_p90':np.percentile(values,[10,50,90]).tolist()} if len(values) else {'count':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    protocol_path=Path('assets/pi05_piper_transition/eggplant_potato/r2_v1/protocol.json')
    protocol=json.loads(protocol_path.read_text());root=Path('Datasets/eggplant_potato_gripper_binary')
    spans=[];scalars={};hashes={}
    for episode in protocol['fit_episodes']:
        path=root/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet'
        hashes[str(episode)]=hashlib.sha256(path.read_bytes()).hexdigest()
        table=pq.read_table(path,columns=['timestamp','subtask','observation.state','action']).to_pydict()
        stamps=np.array(table['timestamp']);labels=table['subtask'];states=np.array(table['observation.state']);actions=np.array(table['action'])
        boundaries=np.array([i for i in range(1,len(labels)) if labels[i]!=labels[i-1]])
        starts=[0,*boundaries.tolist()];ends=[*boundaries.tolist(),len(labels)]
        for start,end in zip(starts,ends,strict=True):
            valid=[i for i in range(start,end) if not len(boundaries) or min(abs(i-boundaries))>3]
            label=labels[start];seconds=(end-start)*float(np.median(np.diff(stamps)))
            entry={'episode':episode,'label':label,'start':start,'end_exclusive':end,'frames':end-start,
                   'seconds':seconds,'D_unmasked_frames_if_active_known':len(valid),'left_width':summary(states[start:end,6]),
                   'right_width':summary(states[start:end,13])}
            if 'grasp' in label.lower():
                joint=13 if 'handle' in label.lower() else 6
                entry['closure_reference_offsets']={}
                for threshold in [.02,.04,.06]:
                    crossings=[i for i in range(1,len(labels)-2) if states[i-1,joint]>=threshold
                               and np.all(states[i:i+3,joint]<threshold) and abs(stamps[i]-stamps[start])<=2.5]
                    nearest=min(crossings,key=lambda i:abs(stamps[i]-stamps[start])) if crossings else None
                    entry['closure_reference_offsets'][str(threshold)]={
                        'frame':nearest,'seconds_after_grasp_label':float(stamps[nearest]-stamps[start]) if nearest is not None else None}
            spans.append(entry)
            scalars.setdefault(label,[]).append(states[start:end][:,[6,13]])
    phases={}
    for label in sorted(scalars):
        rows=[r for r in spans if r['label']==label];state=np.concatenate(scalars[label])
        phases[label]={'episodes':len(rows),'frame_total':len(state),'duration_seconds':summary([r['seconds'] for r in rows]),
                      'phase_frames':summary([r['frames'] for r in rows]),
                      'spans_with_no_unmasked_D_supervision':sum(r['D_unmasked_frames_if_active_known']==0 for r in rows),
                      'total_unmasked_D_frames_if_active_known':sum(r['D_unmasked_frames_if_active_known'] for r in rows),
                      'left_width':summary(state[:,0]),'right_width':summary(state[:,1])}
        if 'grasp' in label.lower():
            phases[label]['closure_reference_offsets_seconds']={t:summary([r['closure_reference_offsets'][t]['seconds_after_grasp_label']
                               for r in rows if r['closure_reference_offsets'][t]['seconds_after_grasp_label'] is not None]) for t in ['0.02','0.04','0.06']}
    # Do not conflate a partially closed right gripper with handle-grasp stage.
    right_partial={label:sum(int(np.count_nonzero((x[:,1]<.04)&(x[:,1]>.005))) for x in values) for label,values in scalars.items()}
    result={'scope':'138 R2 FIT episodes only; original labels and recorded continuous state; no model inference or test access',
            'episodes':len(protocol['fit_episodes']),'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'phases':phases,'right_partial_width_005_to_040_frame_counts':right_partial,'spans':spans,'parquet_sha256':hashes,
            'physical_readiness_reviewed':False,'threshold_scope':'Width crossings are numeric references only, not contact/grasp-success or release-readiness labels.',
            'D_scope':'Maximum non-uncertain supervision opportunities for known active text; actual model-history sampling can provide fewer.'}
    (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['spans','parquet_sha256']},indent=2),flush=True)


if __name__=='__main__':main()
