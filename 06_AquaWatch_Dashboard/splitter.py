"""Adapted from the user's split.py: split a single uploaded recording, preserving timestamps."""
from pathlib import Path
import math
import cv2

MIN_DURATION = 6
PREFERRED_MIN = 8
MAX_DURATION = 10


def calculate_segments(duration):
    if duration < MIN_DURATION:
        raise ValueError(f'Recording is {duration:.1f}s. Upload at least {MIN_DURATION}s of footage.')
    if duration <= MAX_DURATION:
        return 1
    for n in range(math.ceil(duration / MAX_DURATION), math.floor(duration / PREFERRED_MIN) + 1):
        if PREFERRED_MIN <= duration / n <= MAX_DURATION:
            return n
    # For otherwise unworkable lengths, choose the nearest balanced segmentation.
    # E.g. 14s => 7s + 7s; 21s => 7s x 3.
    n = math.ceil(duration / MAX_DURATION)
    if duration / n < MIN_DURATION:
        raise ValueError('Unable to divide video into clips of at least six seconds.')
    return n


def split_video(video_path, output_dir):
    video_path, output_dir = Path(video_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video_path.name}')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
            raise ValueError('Video has invalid frame rate, dimensions, or frame count.')
        duration = frames / fps
        count = calculate_segments(duration)
        segments = []
        for i in range(count):
            start = round(i * frames / count)
            end = round((i + 1) * frames / count)
            path = output_dir / f'{video_path.stem}_part{i+1:03d}.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f'Could not create segment {path.name}')
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            written = 0
            try:
                for _ in range(end-start):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(frame)
                    written += 1
            finally:
                writer.release()
            if written != end-start:
                raise RuntimeError(f'Incomplete segment {path.name}: {written}/{end-start} frames')
            segments.append({'segment':i+1,'filename':path.name,'path':path,
                             'start_seconds':round(start/fps,3),'end_seconds':round(end/fps,3),
                             'duration_seconds':round((end-start)/fps,3)})
        return segments
    finally:
        cap.release()
