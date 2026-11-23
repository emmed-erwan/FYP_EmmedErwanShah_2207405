"""Process new MP4s in inbox exactly once. Run with Task Scheduler at desired intervals."""
from pathlib import Path
import argparse
import json
import shutil
from engine import ROOT, predict_video

p=argparse.ArgumentParser()
p.add_argument('--once',action='store_true',help='Process all unprocessed videos now')
p.add_argument('--threshold',type=float,default=.5)
a=p.parse_args()
if not a.once: p.error('Specify --once. Schedule this command externally.')
inbox=ROOT/'inbox'; inbox.mkdir(exist_ok=True)
processed=ROOT/'processed'; processed.mkdir(exist_ok=True)
for video in sorted(inbox.glob('*.mp4')):
    print(f'Processing {video.name}',flush=True)
    try:
        result,_=predict_video(video,source='scheduled',threshold=a.threshold)
        print(result.to_string(index=False),flush=True)
        destination=processed/video.name
        if destination.exists():
            import uuid
            destination=processed/f'{video.stem}_{uuid.uuid4().hex[:8]}.mp4'
        shutil.move(str(video),str(destination))
    except Exception as e:
        print(f'FAILED {video.name}: {e}',flush=True)
        # Leave failed files for inspection/retry. See failed status in dashboard.
