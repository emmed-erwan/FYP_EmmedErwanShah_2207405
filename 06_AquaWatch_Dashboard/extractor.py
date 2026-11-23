"""Headless MP4 -> coordinate CSV, adapted from the supplied model.py."""
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

HEADERS = ['Video_Name','Frame','Fish_ID','BBox_X1','BBox_Y1','BBox_X2','BBox_Y2','Eye_X','Eye_Y','Tail_X','Tail_Y','Center_X','Center_Y']

def extract_video(video_path, output_csv, detector_path, pose_path, annotated_path=None):
    video_path, output_csv = Path(video_path), Path(output_csv)
    if not video_path.is_file(): raise FileNotFoundError(video_path)
    for weight in (detector_path, pose_path):
        if not Path(weight).is_file(): raise FileNotFoundError(f'Missing YOLO weights: {weight}')
    detector = YOLO(str(detector_path))  # fresh tracker per clip
    pose = YOLO(str(pose_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {video_path}')
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    writer_video = None
    if annotated_path:
        annotated_path = Path(annotated_path)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        writer_video = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (854,480))
    frames = 0; detections = 0
    try:
        with output_csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(HEADERS)
            while True:
                ok, frame = cap.read()
                if not ok: break
                frames += 1
                frame = cv2.resize(frame, (854,480))
                result = detector.track(frame, persist=True, tracker='botsort.yaml', conf=.55, iou=.50, verbose=False)[0]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    ids = result.boxes.id.int().cpu().tolist()
                    for fid, (x1,y1,x2,y2) in zip(ids,boxes):
                        px1,py1,px2,py2 = max(0,x1-15),max(0,y1-15),min(854,x2+15),min(480,y2+15)
                        crop = frame[py1:py2,px1:px2]
                        if crop.size == 0: continue
                        pres = pose.predict(crop, conf=.25, verbose=False)[0]
                        eye = tail = None
                        if pres.keypoints is not None and len(pres.keypoints):
                            kpts = pres.keypoints.xy.cpu().numpy()[0]
                            if len(kpts)>=2 and all(kpts[0]>0) and all(kpts[1]>0):
                                eye = (int(kpts[0][0]+px1),int(kpts[0][1]+py1))
                                tail = (int(kpts[1][0]+px1),int(kpts[1][1]+py1))
                        cx,cy = ((eye[0]+tail[0])//2,(eye[1]+tail[1])//2) if eye and tail else ((x1+x2)//2,(y1+y2)//2)
                        w.writerow([video_path.name,frames,fid,x1,y1,x2,y2,*(eye or (-1,-1)),*(tail or (-1,-1)),cx,cy])
                        detections += 1
                        if writer_video:
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                            cv2.putText(frame,f'ID:{fid}',(x1,max(20,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),2)
                if writer_video: writer_video.write(frame)
    finally:
        cap.release()
        if writer_video: writer_video.release()
    if detections==0: raise ValueError(f'No tracked fish detected in {video_path.name}')
    return {'frames':frames,'detections':detections}
