import collections
import json
from pathlib import Path
import numpy as np
from openpi.training.subtask_transition import BoundarySampler, read_jsonl, sampled_rows

root=Path('/media/raid/workspace/surongpeng/ws_lixing/agentic-openpi')
out=Path('/tmp/transition_review_20260908');out.mkdir(exist_ok=True)
assets=root/'checkpoints/pi05_piper_transition/r1_seed42/transition_assets'
rows=read_jsonl(assets/'frames_train.jsonl')
sampler=BoundarySampler(rows,batch_size=2,accumulation=2,steps=1000,seed=42,world_size=8)
draws=np.concatenate([sampler.global_indices(s).flatten() for s in range(1000)]).tolist()
natural=collections.Counter(r['label'] for r in rows)
exposure=collections.Counter(rows[i]['label'] for i in draws)
events=read_jsonl(assets/'boundary_index.jsonl')
val=read_jsonl(assets/'frames_val.jsonl')
observability={}
for phase in [0.,.255,.509]:
 selected=sampled_rows(val,.764,phase)
 by_ep=collections.defaultdict(list)
 for r in selected:by_ep[r['episode']].append(r)
 items=[]
 for e in events:
  if e['split']!='val':continue
  seen=[r for r in by_ep[e['episode']] if e['t_label']<=r['timestamp']<e['new_end_time']]
  items.append({'event_id':e['event_id'],'observations_inside_new_stage':len(seen),'duration':e['new_end_time']-e['t_label']})
 observability[str(phase)]={'zero':sum(x['observations_inside_new_stage']==0 for x in items),'one':sum(x['observations_inside_new_stage']==1 for x in items),'at_least_two':sum(x['observations_inside_new_stage']>=2 for x in items),'events_detail':items}
result={'sampling':{'draws':len(draws),'unique_frames':len(set(draws)),'duplicate_draw_fraction':1-len(set(draws))/len(draws),'boundary_draws':sum(rows[i]['boundary_event'] is not None for i in draws),'class_exposure':[{'label':k,'natural_count':natural[k],'natural_fraction':natural[k]/len(rows),'sampled_count':exposure[k],'sampled_fraction':exposure[k]/len(draws)} for k in natural]},'low_rate_label_observation_opportunities':observability,'annotation_review_counts':dict(collections.Counter(e['review_status'] for e in events))}
prior=root/'logs/pi05_subtask_stage1/m3_native_dense_val'
r0=[r for p in sorted(prior.glob('semantics_rank_*.json')) for r in json.loads(p.read_text())]
r1=json.loads((root/'checkpoints/pi05_piper_transition/r1_seed42/predictions_001000.json').read_text())
conf={}
for name,preds in [('R0',r0),('R1_1000',r1)]:
 c=collections.defaultdict(collections.Counter)
 for r in preds:c[r['label']][r['prediction']]+=1
 conf[name]={k:dict(v) for k,v in c.items()}
result['validation_confusion']=conf
(out/'training_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'sampling':result['sampling'],'annotation_review_counts':result['annotation_review_counts'],'observability':{k:{a:b for a,b in v.items() if a!='events_detail'} for k,v in observability.items()},'handle_grasp_confusion':{k:v.get('Grasp the handle of the lid') for k,v in conf.items()}},ensure_ascii=False,indent=2))

# Concrete training examples for visual phase ambiguity; this is inspection, not a readiness annotation.
import av
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
ds=root/'Datasets/eggplant_potato_gripper_binary'
table=pq.read_table(ds/'data/chunk-000/episode_000000.parquet').to_pydict()
stage_indices={}
for i,label in enumerate(table['subtask']):stage_indices.setdefault(label,[]).append(i)
chosen=[('reach the handle of the lid',-8),('Grasp the handle of the lid',10),('Put the lid on the box',10)]
sheet=Image.new('RGB',(1440,390*3),'white');draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
meta=[]
for row,(label,offset) in enumerate(chosen):
 ids=stage_indices[label];idx=ids[offset] if offset<len(ids) else ids[len(ids)//2]
 t=table['timestamp'][idx];meta.append({'label':label,'frame':idx,'time':t,'state':table['observation.state'][idx]})
 draw.text((5,row*390+3),f'TRAIN episode 0 | frame {idx} | {t:.2f}s | original label: {label}',font=font,fill='black')
 for col,view in enumerate(['cam_high','cam_left_wrist','cam_right_wrist']):
  with av.open(str(ds/f'videos/chunk-000/observation.images.{view}/episode_000000.mp4')) as video:
   for j,frame in enumerate(video.decode(video=0)):
    if j==idx:
     sheet.paste(frame.to_image().resize((480,360)),(col*480,row*390+30));break
sheet.save(out/'training_phase_examples.jpg',quality=94)
(out/'training_phase_examples.json').write_text(json.dumps(meta,indent=2))
