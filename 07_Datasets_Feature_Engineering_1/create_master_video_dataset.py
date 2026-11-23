"""Build four-fish, ten-observation, 24-permutation video-level datasets.

Usage:
 python create_master_video_dataset.py --input fleeing_behaviour_cleaned.csv --output-dir fleeing_master
 python create_master_video_dataset.py --image-width 640 --image-height 480

Each fish is sampled independently at evenly spaced frame positions across its
own observed track. Movement features are computed from consecutive actual detections first, then
sampled with linear interpolation and nearest-valid edge filling. The first
sampled movement uses the nearest valid motion measurement, not forced zero. Pairwise statistics use real, simultaneous detections
only; missing group statistics are set to 0 and explicitly flagged.

IMPORTANT: Permutations are augmentations of the SAME video. Split by
Source_Video before ML training. Camera dimensions are preferable for geometry
normalization; otherwise observed maximum coordinates are proxy denominators.
"""
import argparse
from itertools import combinations, permutations
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ('Change_Area', 'DCenter_X', 'DCenter_Y', 'Vector_X', 'Vector_Y')
REQUIRED = {'Video_Name','Frame','Fish_ID','BBox_X1','BBox_Y1','BBox_X2','BBox_Y2','Center_X','Center_Y'}
COORDS = ('BBox_X1','BBox_Y1','BBox_X2','BBox_Y2','Center_X','Center_Y')


def pairwise_stats(video):
    distances = []
    for _, frame in video.groupby('Frame', sort=False):
        pts = frame[['Center_X','Center_Y']].dropna().to_numpy(dtype=float)
        for a, b in combinations(pts, 2):
            distances.append(float(np.linalg.norm(a-b)))
    if not distances:
        return 0.0, 0.0, 0
    return float(np.mean(distances)), float(np.std(distances)), len(distances)


def make_sequence(track, observations):
    """Sample nearest/interpolated VALID movement features from one fish track.

    Movement features are computed on consecutive *original* detections before
    sampling, so synthetic coordinate plateaus do not create artificial zeros.
    At the first observation, borrow the closest valid movement measurement.
    """
    track = track.sort_values('Frame').drop_duplicates('Frame').copy()
    frame = track['Frame'].to_numpy(dtype=float)
    positions = np.linspace(frame[0], frame[-1], observations)
    cx = track['Center_X'].to_numpy(dtype=float)
    cy = track['Center_Y'].to_numpy(dtype=float)
    area = ((track['BBox_X2'] - track['BBox_X1']).clip(lower=0) *
            (track['BBox_Y2'] - track['BBox_Y1']).clip(lower=0)).to_numpy(dtype=float)
    dt = np.diff(frame)
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.r_[np.nan, np.diff(cx)]
        dy = np.r_[np.nan, np.diff(cy)]
        da = np.r_[np.nan, np.diff(area)]
        vx = np.r_[np.nan, np.diff(cx) / dt]
        vy = np.r_[np.nan, np.diff(cy) / dt]
    original = dict(zip(FEATURES, (da, dx, dy, vx, vy)))
    sequence = {}
    for feature, values in original.items():
        valid = np.isfinite(values)
        if not valid.any():
            # Single-observation tracks have no measurable motion; track
            # selection normally excludes these. Keep explicit fallback.
            sequence[feature] = np.zeros(observations, dtype=float)
        else:
            sequence[feature] = np.interp(positions, frame[valid], values[valid])
    return sequence, positions


def build(input_path, output_dir, observations=10, fish_count=4,
          label='Fleeing', image_width=None, image_height=None):
    if fish_count != 4:
        raise ValueError('This pipeline uses exactly four fish (24 permutations).')
    if observations < 2:
        raise ValueError('observations must be >= 2')
    if (image_width is None) != (image_height is None):
        raise ValueError('Specify both image dimensions or neither.')
    df = pd.read_csv(input_path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    for col in REQUIRED - {'Video_Name'}:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Video_Name','Frame','Fish_ID']).copy()
    df = df.sort_values(['Video_Name','Fish_ID','Frame'], kind='stable')
    duplicates = int(df.duplicated(['Video_Name','Fish_ID','Frame']).sum())
    df = df.drop_duplicates(['Video_Name','Fish_ID','Frame'], keep='first')
    output_rows = []
    skipped = []
    for name, video in df.groupby('Video_Name', sort=True):
        # Select longest valid tracks, not those that happen to appear at
        # arbitrary common video-wide frame positions.
        valid = video.dropna(subset=list(COORDS))
        coverage = valid.groupby('Fish_ID').Frame.nunique().sort_values(ascending=False)
        ids = list(coverage.index[:fish_count])
        if len(ids) < fish_count:
            skipped.append((name, len(ids)))
            continue
        mean_dist, std_dist, pair_count = pairwise_stats(video)
        if image_width is not None:
            width, height = float(image_width), float(image_height)
            norm_source = 'camera_dimensions'
        else:
            width = float(max(1, video[['BBox_X1','BBox_X2','Center_X']].max().max()))
            height = float(max(1, video[['BBox_Y1','BBox_Y2','Center_Y']].max().max()))
            norm_source = 'observed_coordinate_extent_proxy'
        if width <= 0 or height <= 0:
            raise ValueError('Image dimensions must be positive')
        sequences = [make_sequence(video.loc[video.Fish_ID == fid], observations) for fid in ids]
        base = {'Source_Video':name, 'Video_Name':name, 'Video_Label':label,
                'Original_Frame_Count':int(video.Frame.nunique()),
                'Detected_Fish_Count':int(video.Fish_ID.nunique()),
                'Video_Avg_Pairwise_Distance':mean_dist,
                'Video_Std_Pairwise_Distance':std_dist,
                'Video_Pairwise_Observation_Count':pair_count,
                'Pairwise_Stats_Imputed':int(pair_count == 0),
                'Normalization_Width':width, 'Normalization_Height':height,
                'Normalization_Source':norm_source}
        for permutation_id, order in enumerate(permutations(range(fish_count)), 1):
            row = dict(base)
            row['Permutation_ID'] = permutation_id
            for slot, ix in enumerate(order, 1):
                seq, positions = sequences[ix]
                row[f'ID{slot}_Original_Fish_ID'] = ids[ix]
                row[f'ID{slot}_Sampled_Source_Frames'] = '|'.join(f'{f:.3f}' for f in positions)
                for j in range(observations):
                    for feature in FEATURES:
                        row[f'ID{slot}_F{j}_{feature}'] = float(seq[feature][j])
            output_rows.append(row)
    if not output_rows:
        raise ValueError('No videos contain four valid fish tracks.')
    result = pd.DataFrame(output_rows)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / 'master_fleeing_non_normalized.csv'
    norm_path = out_dir / 'master_fleeing_normalized.csv'
    result.to_csv(raw_path, index=False)
    norm = result.copy()
    for feature in FEATURES:
        cols = [c for c in norm.columns if c.startswith('ID') and c.endswith('_' + feature)]
        if feature == 'Change_Area':
            scale = norm.Normalization_Width * norm.Normalization_Height
        elif feature in ('DCenter_X','Vector_X'):
            scale = norm.Normalization_Width
        else:
            scale = norm.Normalization_Height
        norm[cols] = norm[cols].div(scale, axis=0)
    diagonal = np.hypot(norm.Normalization_Width, norm.Normalization_Height)
    for col in ('Video_Avg_Pairwise_Distance','Video_Std_Pairwise_Distance'):
        norm[col] = norm[col] / diagonal
    norm.to_csv(norm_path, index=False)
    print(f'Input observations: {len(df):,}; duplicate rows dropped: {duplicates}')
    print(f'Videos: {result.Source_Video.nunique()}, skipped: {len(skipped)}, rows: {len(result):,}')
    print(f'Videos without pairwise distances: {result.groupby("Source_Video").Pairwise_Stats_Imputed.first().sum()}')
    print(f'Missing feature cells: {int(result.filter(regex=r"^ID[1-4]_F\\d+_(?:Change_Area|DCenter_X|DCenter_Y|Vector_X|Vector_Y)$").isna().sum().sum())}')
    print(f'Non-normalized: {raw_path}\nNormalized: {norm_path}')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='fleeing_behaviour_cleaned.csv')
    parser.add_argument('--output-dir', default='fleeing_master')
    parser.add_argument('--observations', type=int, default=10)
    parser.add_argument('--fish-count', type=int, default=4)
    parser.add_argument('--label', default='Fleeing')
    parser.add_argument('--image-width', type=int)
    parser.add_argument('--image-height', type=int)
    args = parser.parse_args()
    build(args.input, args.output_dir, args.observations, args.fish_count,
          args.label, args.image_width, args.image_height)
