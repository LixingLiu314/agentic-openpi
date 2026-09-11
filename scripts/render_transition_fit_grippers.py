"""Recorded FIT-only contact sheets for the scalar grasp audit; no model inference."""
import argparse
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.audit/'report.json').read_text());root=Path('Datasets/eggplant_potato_gripper_binary')
    directory=a.audit/'visuals';directory.mkdir(exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    index=[]
    for sample in report['examples_for_visual_review']:
        ep=sample['episode'];start=sample['start'];end=sample['end_exclusive']
        frames=[start-3,start,min(end-1,start+15),end-1];images={}
        data=pq.read_table(root/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet',columns=['subtask','observation.state']).to_pydict()
        for view in ['cam_high','cam_left_wrist','cam_right_wrist']:
            with av.open(str(root/f'videos/chunk-{ep//1000:03d}/observation.images.{view}/episode_{ep:06d}.mp4')) as video:
                for frame_id,decoded in enumerate(video.decode(video=0)):
                    if frame_id in frames:images[(view,frame_id)]=decoded.to_image().resize((320,240))
                    if frame_id>=max(frames):break
        canvas=Image.new('RGB',(960,1180),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f"FIT episode {ep} | acting hand {sample['acting_hand']} | {sample['label']}",font=font,fill='black')
        draw.text((8,28),'Original labels and recorded widths; this sheet does not certify successful grasp.',font=font,fill='black')
        for row,frame in enumerate(frames):
            state=data['observation.state'][frame];y=58+row*278
            caption=f"f{frame} | {data['subtask'][frame]} | L {state[6]:.5f} / R {state[13]:.5f}"
            draw.text((8,y),caption,font=font,fill='black')
            for col,view in enumerate(['cam_high','cam_left_wrist','cam_right_wrist']):canvas.paste(images[(view,frame)],(320*col,y+25))
        name=f"episode_{ep:06d}_{sample['acting_hand']}.jpg";canvas.save(directory/name,quality=95)
        index.append({'file':name,'episode':ep,'frames':frames,'acting_hand':sample['acting_hand'],'label':sample['label']})
    (directory/'index.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(index))


if __name__=='__main__':main()
