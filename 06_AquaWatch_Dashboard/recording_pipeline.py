"""One long recording -> segments -> YOLO extraction -> 202 features -> RF -> timeline."""
from pathlib import Path
from datetime import datetime
import uuid
import pandas as pd
from splitter import split_video
from engine import ROOT, predict_coordinates, save_event


def analyse_recording(video_path, source='manual', on_progress=None):
    from extractor import extract_video
    video_path = Path(video_path)
    run_id = uuid.uuid4().hex[:12]
    run_dir = ROOT / 'outputs' / f'recording_{run_id}'
    clips_dir = run_dir / 'segments'
    clips = split_video(video_path, clips_dir)
    rows = []
    for index, clip in enumerate(clips, 1):
        if on_progress:
            on_progress(index-1, len(clips), f'Analysing segment {index}/{len(clips)}')
        coord_path = run_dir / 'coordinates' / (Path(clip['filename']).stem + '.csv')
        try:
            extract_video(clip['path'], coord_path, ROOT/'weights'/'best_combine.pt', ROOT/'weights'/'bestv7.pt')
            predictions, _ = predict_coordinates(coord_path, source=source, display_name=clip['filename'], threshold=0.50)
            pred = predictions.iloc[0]
            label = int(pred['Predicted_Label'])
            probability = float(pred['Fleeing_Probability'])
            status, message = 'completed', 'Fleeing detected' if label else 'No fleeing detected'
        except Exception as exc:
            label, probability, status, message = None, None, 'failed', str(exc)
            save_event({'id':uuid.uuid4().hex,'recorded_at':datetime.now().astimezone().isoformat(timespec='seconds'),
                        'source':source,'video_name':clip['filename'],'predicted_label':None,
                        'fleeing_probability':None,'status':'failed','message':message,'output_path':str(run_dir)})
        rows.append({'Recording':video_path.name,'Segment':clip['segment'],
                     'Start_s':clip['start_seconds'],'End_s':clip['end_seconds'],
                     'Duration_s':clip['duration_seconds'],'Clip':clip['filename'],
                     'Status':status,'Prediction':('Fleeing' if label else 'Non-Fleeing') if status=='completed' else 'Not analysed',
                     'Fleeing_Score':probability,'Message':message})
    result = pd.DataFrame(rows)
    result.to_csv(run_dir/'timeline.csv',index=False)
    if on_progress:
        on_progress(len(clips),len(clips),'Analysis finished')
    return result, run_dir
