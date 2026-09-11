"""Render actual sampled history/current images and S output for update one."""
import argparse
import json
from pathlib import Path
import av
from PIL import Image,ImageDraw,ImageFont


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    sample=json.loads((args.directory/'sample.json').read_text())
    root=Path('Datasets/eggplant_potato_gripper_binary');ep=sample['row']['episode']
    frames=sample['history']+[sample['row']['frame']];wanted={i for i in frames if i is not None}
    views=['cam_high','cam_left_wrist','cam_right_wrist'];images={}
    for view in views:
        with av.open(str(root/f'videos/chunk-{ep//1000:03d}/observation.images.{view}/episode_{ep:06d}.mp4')) as video:
            for i,frame in enumerate(video.decode(video=0)):
                if i in wanted:images[(view,i)]=frame.to_image().resize((480,360))
                if i>=max(wanted):break
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    canvas=Image.new('RGB',(1440,1750),'white');draw=ImageDraw.Draw(canvas)
    for row,frame in enumerate(frames):
        title=f"{'CURRENT' if row==3 else 'PAST'} | episode {ep} | frame {frame}" if frame is not None else 'PAST | masked: no available observation'
        draw.text((8,row*390+3),title,font=font,fill='black')
        for col,view in enumerate(views):canvas.paste(images.get((view,frame),Image.new('RGB',(480,360),'#cccccc')),(col*480,row*390+30))
    lines=[f"Update 1 | prior model text: {sample['active_text'] or '<empty>'}",
           f"Training target (sidecar only): {sample['row']['label']}",f"Generated output: {sample['prediction']}",
           f"Keep/advance/reidentify annotation-proxy probabilities: {sample['completion_probabilities']}"]
    for i,line in enumerate(lines):draw.text((8,1570+i*36),line,font=font,fill='black')
    canvas.save(args.directory/'input_output.png')


if __name__=='__main__':main()
