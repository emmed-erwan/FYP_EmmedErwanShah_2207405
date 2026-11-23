import os
import math
import cv2
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

VIDEOS_DIR = r"feeding_pipeline\videos"

TRACKING_DIR = r"feeding_pipeline\csv"

CLIPS_DIR = r"feeding_pipeline\master_clips"

CLIP_TRACKING_DIR = (
    r"feeding_pipeline\master_clip_tracking"
)

MASTER_MANIFEST = (
    r"feeding_pipeline\master_clips\master_clip_manifest.csv"
)

CLIP_LENGTH = 30


SOURCE_VIDEOS = [
    "feeding_01.mp4",
    "feeding_02.mp4",
    "feeding_03.mp4",
    "feeding_04.mp4",
    "feeding_05.mp4",
    "feeding_06.mp4",
    "feeding_07.mp4",
    "feeding_08.mp4",
    "feeding_09.mp4",
    "feeding_10.mp4",
    "feeding_11.mp4",
    "feeding_12.mp4",
    "feeding_13.mp4",
    "feeding_14.mp4",
    "feeding_15.mp4",
    "feeding_16.mp4",
    "feeding_17.mp4",
    "feeding_18.mp4",
    "feeding_19.mp4",
    "feeding_20.mp4",
    "feeding_21.mp4",
    "feeding_22.mp4",
    "feeding_23.mp4",
    "feeding_24.mp4",
    "feeding_25.mp4",
]


os.makedirs(
    CLIPS_DIR,
    exist_ok=True
)

os.makedirs(
    CLIP_TRACKING_DIR,
    exist_ok=True
)


# ============================================================
# LOAD EXISTING feeding_01 LABELS
# ============================================================

existing_labels = {}

feeding01_manifest = (
    r"feeding_pipeline\all_fixed_clips\all_fixed_clips_manifest.csv"
)

if os.path.exists(feeding01_manifest):

    old_manifest = pd.read_csv(
        feeding01_manifest
    )

    for _, row in old_manifest.iterrows():

        if pd.notna(
            row["video_label"]
        ):

            existing_labels[
                int(row["window_id"])
            ] = int(
                row["video_label"]
            )


# ============================================================
# PROCESS ALL SOURCE VIDEOS
# ============================================================

manifest_rows = []


for source_video in SOURCE_VIDEOS:

    source_stem = os.path.splitext(
        source_video
    )[0]

    video_path = os.path.join(
        VIDEOS_DIR,
        source_video
    )

    tracking_path = os.path.join(
        TRACKING_DIR,
        f"{source_stem}_tracking_pose.csv"
    )


    print()
    print("========================================")
    print("Processing:", source_video)
    print("========================================")


    if not os.path.exists(video_path):
        print("Missing video:", video_path)
        continue

    if not os.path.exists(tracking_path):
        print("Missing tracking CSV:", tracking_path)
        continue


    tracking = pd.read_csv(
        tracking_path
    )


    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        print("Could not open:", video_path)
        continue


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    total_clips = math.ceil(
        total_frames / CLIP_LENGTH
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    for window_id in range(
        total_clips
    ):

        start_frame = (
            window_id
            *
            CLIP_LENGTH
        )

        end_frame = min(
            start_frame
            +
            CLIP_LENGTH
            -
            1,
            total_frames - 1
        )


        clip_name = (
            f"{source_stem}"
            f"_window_{window_id:03d}.mp4"
        )

        clip_path = os.path.join(
            CLIPS_DIR,
            clip_name
        )


        # ----------------------------------------------------
        # EXTRACT VIDEO CLIP
        # ----------------------------------------------------

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        writer = cv2.VideoWriter(
            clip_path,
            fourcc,
            fps,
            (width, height)
        )

        frames_written = 0


        for frame_number in range(
            start_frame,
            end_frame + 1
        ):

            ok, frame = cap.read()

            if not ok:
                break

            writer.write(
                frame
            )

            frames_written += 1


        writer.release()


        # ----------------------------------------------------
        # SLICE TRACKING CSV
        # ----------------------------------------------------

        clip_tracking = tracking[
            (tracking["frame"] >= start_frame)
            &
            (tracking["frame"] <= end_frame)
        ].copy()


        clip_tracking[
            "source_frame"
        ] = clip_tracking[
            "frame"
        ]


        clip_tracking[
            "frame"
        ] = (
            clip_tracking["frame"]
            -
            start_frame
        )


        clip_tracking[
            "timestamp_sec"
        ] = (
            clip_tracking["frame"]
            /
            fps
        )


        clip_tracking[
            "video"
        ] = clip_name


        clip_tracking_name = (
            f"{source_stem}"
            f"_window_{window_id:03d}"
            f"_tracking_pose.csv"
        )

        clip_tracking_path = os.path.join(
            CLIP_TRACKING_DIR,
            clip_tracking_name
        )


        clip_tracking.to_csv(
            clip_tracking_path,
            index=False
        )


        # ----------------------------------------------------
        # TRACKING SUMMARY
        # ----------------------------------------------------

        fish_ids = (
            clip_tracking[
                "track_id"
            ].nunique()
            if len(clip_tracking) > 0
            else 0
        )

        observed_frames = (
            clip_tracking[
                "frame"
            ].nunique()
            if len(clip_tracking) > 0
            else 0
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        video_label = ""

        label_status = "unlabelled"


        # Keep labels already reviewed for feeding_01.
        if source_stem == "feeding_01":

            if window_id in existing_labels:

                video_label = (
                    existing_labels[
                        window_id
                    ]
                )

                label_status = (
                    "previously_reviewed"
                )


        # ----------------------------------------------------
        # CLIP QUALITY
        # ----------------------------------------------------

        if frames_written < CLIP_LENGTH:

            clip_status = (
                "partial_clip"
            )

        else:

            clip_status = (
                "full_clip"
            )


        manifest_rows.append({

            "source_video":
                source_video,

            "clip_name":
                clip_name,

            "window_id":
                window_id,

            "start_frame":
                start_frame,

            "end_frame":
                (
                    start_frame
                    +
                    frames_written
                    -
                    1
                ),

            "frames_written":
                frames_written,

            "start_time_sec":
                start_frame / fps,

            "end_time_sec":
                (
                    start_frame
                    +
                    frames_written
                ) / fps,

            "fish_ids":
                fish_ids,

            "observed_frames":
                observed_frames,

            "video_label":
                video_label,

            "label_status":
                label_status,

            "clip_status":
                clip_status,

        })


    cap.release()


# ============================================================
# SAVE MASTER MANIFEST
# ============================================================

manifest = pd.DataFrame(
    manifest_rows
)


manifest.to_csv(
    MASTER_MANIFEST,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("MASTER CLIP DATASET PREPARED")
print("========================================")

print()
print(
    "Total clips:",
    len(manifest)
)

print()

print(
    "Clips per source video:"
)

print(
    manifest.groupby(
        "source_video"
    )
    .size()
    .to_string()
)

print()

print(
    "Clip status:"
)

print(
    manifest[
        "clip_status"
    ]
    .value_counts()
    .to_string()
)

print()

print(
    "Labels currently available:"
)

print(
    manifest[
        "video_label"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print()

print(
    "Master manifest:",
    MASTER_MANIFEST
)

print(
    "Video clips:",
    CLIPS_DIR
)

print(
    "Clip tracking CSVs:",
    CLIP_TRACKING_DIR
)