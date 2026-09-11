import collections
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

p=Path(__file__).parent
source=Path(sys.argv[1]) if len(sys.argv)>1 else p
r=json.loads((source/'run.json').read_text(encoding='utf-8'))
q=r['queries'];v=r['video'];start=v['start_monotonic']
segments=[]
for x in q:
    if not segments or segments[-1]['subtask']!=x['subtask']:
        segments.append({'subtask':x['subtask'],'first_query':x['query'],'last_query':x['query'],
                         'response_start':x['response_monotonic']-start,'observation_start':x['observation']['snapshot_monotonic']-start,
                         'count':0,'published_steps':0})
    segment=segments[-1]
    segment['last_query']=x['query'];segment['count']+=1
    segment['published_steps']+=len(x['action_step_times'])
for i,s in enumerate(segments):
    s['response_end']=segments[i+1]['response_start'] if i+1<len(segments) else v['encoded_seconds']
    s['duration']=s['response_end']-s['response_start']
dt=np.diff([x['observation']['snapshot_monotonic'] for x in q])
latency=np.array([x['roundtrip_ms'] for x in q])
summary={'server':r['server'],'status':r['status'],'segments':segments,
         'counts':dict(collections.Counter(x['subtask'] for x in q)),
         'latency_ms_p50_p95_max':np.percentile(latency,[50,95,100]).tolist(),
         'request_spacing_s_p50_p95_max':np.percentile(dt,[50,95,100]).tolist(),
         'subtask_status_counts':dict(collections.Counter(x['subtask_status'] for x in q)),
         'max_sensor_age':{k:max(x['observation']['sensor_age_seconds'].get(k,0) for x in q) for k in q[0]['observation']['sensor_age_seconds']},
         'video':{k:v[k] for k in ['encoded_seconds','frame_count','repeated_timing_frames','max_capture_gap_seconds']}}
compact=[]
for x in q:
    a=np.asarray(x['raw_actions']);state=np.asarray(x['state']);n=len(x['action_step_times'])
    compact.append({'query':x['query'],'t_input':x['observation']['snapshot_monotonic']-start,
                    't_response':x['response_monotonic']-start,'subtask':x['subtask'],'score':x['subtask_score'],
                    'grippers_state':state[[6,13]].tolist(),'first_gripper_targets':a[0,[6,13]].tolist(),
                    'first15_gripper_min':a[:15,[6,13]].min(0).tolist(),'first15_gripper_max':a[:15,[6,13]].max(0).tolist(),
                    'published_steps':n,'joints_state':state.tolist(),
                    'first_command_delay_ms':(x['action_step_times'][0]-x['time'])*1000 if n else None})
summary['queries_compact']=compact
(p/'trial_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['queries_compact','server']},ensure_ascii=False,indent=2))
for x in compact:
    if x['query']%10==1 or any(abs(x['query']-s['first_query'])<=2 for s in segments):
        print(json.dumps({k:v for k,v in x.items() if k!='joints_state'},ensure_ascii=False))

times=sorted(set([0.,5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.]+[round(s['response_start']+.1,2) for s in segments]))
cap=cv2.VideoCapture(str(source/'video.mp4'))
font_path='C:/Windows/Fonts/arial.ttf' if sys.platform=='win32' else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(font_path,22)
for page in range((len(times)+5)//6):
    canvas=Image.new('RGB',(1440, (288+32)*min(6,len(times)-page*6)), 'white')
    draw=ImageDraw.Draw(canvas)
    for slot,t in enumerate(times[page*6:page*6+6]):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
        ok,bgr=cap.read()
        if not ok:continue
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(rgb)
        image.save(p/f'frame_{t:06.2f}.jpg',quality=95)
        # Two columns? Preserve all three views across width, omit black banner only.
        panel=image.resize((720,288))
        preceding=[x for x in compact if x['t_response']<=t]
        label=preceding[-1]['subtask'] if preceding else 'no returned prediction yet'
        draw.text((5,slot*320+2),f't={t:.2f}s | {label}',font=font,fill='black')
        canvas.paste(panel,(0,slot*320+32))
        # Enlarged front camera for the scene evidence, alongside original three views.
        front=image.crop((0,92,480,452)).resize((384,288))
        wrist=image.crop((960,92,1440,452)).resize((336,252))
        canvas.paste(front,(720,slot*320+32))
        canvas.paste(wrist,(1104,slot*320+32))
    canvas.save(p/f'contact_{page+1}.jpg',quality=92)
cap.release()
