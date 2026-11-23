import os
import cv2
import csv
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Configuration & Folder Setup
# ---------------------------------------------------------

DETECTOR_PATH = "best_combine.pt"
POSE_PATH = "bestv7.pt"

# Put all input .mp4 files in this folder
VIDEO_FOLDER = r"C:\Users\user\Desktop\UTP\MAY26\training\videos_test"

# Output folders
VIDEO_DIR = "output_video_test"
TEXT_DIR = "output_text_test"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Find all MP4 videos
video_files = [
    os.path.join(VIDEO_FOLDER, f)
    for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith(".mp4")
]

video_files.sort()

if not video_files:
    raise RuntimeError(f"No MP4 files found in {VIDEO_FOLDER}")

# ---------------------------------------------------------
# 2. Model Confidence / NMS Settings
# ---------------------------------------------------------

DETECTOR_CONF = 0.55
DETECTOR_IOU = 0.50
POSE_CONF = 0.25

# ---------------------------------------------------------
# 3. Video Settings
# ---------------------------------------------------------

RESIZE_WIDTH = 854
RESIZE_HEIGHT = 480

# One CSV contains all videos. Video_Name identifies each row.
OUTPUT_DATA_PATH = os.path.join(
    TEXT_DIR,
    "test_coordinates.csv"
)

# ---------------------------------------------------------
# 4. Load Models
# ---------------------------------------------------------

print("Loading Models...")

detector = YOLO(DETECTOR_PATH)
pose_model = YOLO(POSE_PATH)

# ---------------------------------------------------------
# 5. CSV Setup
# ---------------------------------------------------------

data_file = open(
    OUTPUT_DATA_PATH,
    mode="w",
    newline=""
)

csv_writer = csv.writer(data_file)

csv_writer.writerow([
        "Video_Name",
        "Frame",
        "Fish_ID",
        "BBox_X1",
        "BBox_Y1",
        "BBox_X2",
        "BBox_Y2",
        "Eye_X",
        "Eye_Y",
        "Tail_X",
        "Tail_Y",
        "Center_X",
    "Center_Y"
])

# ---------------------------------------------------------
# 6. Process Each Video
# ---------------------------------------------------------

for video_path in video_files:

    # Fresh tracker state for every independent test clip.
    detector = YOLO(DETECTOR_PATH)

    video_name = os.path.basename(video_path)
    video_base_name = os.path.splitext(video_name)[0]

    print("-----------------------------------------------")
    print(f"Processing: {video_name}")
    print("-----------------------------------------------")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        continue

    width = RESIZE_WIDTH
    height = RESIZE_HEIGHT

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_video_path = os.path.join(
        VIDEO_DIR,
        f"{video_base_name}_coordinates.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0
    stopped_by_user = False

    # -----------------------------------------------------
    # 7. Process Video Frame by Frame
    # -----------------------------------------------------

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Resize frame
        frame = cv2.resize(
            frame,
            (RESIZE_WIDTH, RESIZE_HEIGHT)
        )

        # -------------------------------------------------
        # Fish Detection + BoT-SORT Tracking
        # -------------------------------------------------

        det_results = detector.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=DETECTOR_CONF,
            iou=DETECTOR_IOU,
            verbose=False
        )

        result = det_results[0]

        if result.boxes is None or len(result.boxes) == 0:
            out.write(frame)
            cv2.imshow("Fish Coordinate Extraction", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stopped_by_user = True
                break

            continue

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
            .astype(int)
        )

        if result.boxes.id is not None:
            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )
        else:
            # Untracked detections cannot be assigned reliable Fish_ID values.
            continue

        # -------------------------------------------------
        # Process Each Detected Fish
        # -------------------------------------------------

        for track_id, (x1, y1, x2, y2) in zip(
            track_ids,
            boxes
        ):

            # Add padding around bounding box
            pad = 15

            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(width, x2 + pad)
            crop_y2 = min(height, y2 + pad)

            fish_crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            if fish_crop.size == 0:
                continue

            # -------------------------------------------------
            # Pose Detection
            # -------------------------------------------------

            pose_results = pose_model.predict(
                fish_crop,
                conf=POSE_CONF,
                verbose=False
            )

            eye_global = None
            tail_global = None

            if (
                pose_results[0].keypoints is not None
                and len(pose_results[0].keypoints) > 0
            ):

                kpts_crop = (
                    pose_results[0]
                    .keypoints
                    .xy
                    .cpu()
                    .numpy()[0]
                )

                # Keypoint 0 = Eye
                # Keypoint 1 = Tail
                if len(kpts_crop) >= 2:

                    eye_crop = kpts_crop[0]
                    tail_crop = kpts_crop[1]

                    if (
                        eye_crop[0] > 0
                        and eye_crop[1] > 0
                        and tail_crop[0] > 0
                        and tail_crop[1] > 0
                    ):

                        eye_global = (
                            int(eye_crop[0] + crop_x1),
                            int(eye_crop[1] + crop_y1)
                        )

                        tail_global = (
                            int(tail_crop[0] + crop_x1),
                            int(tail_crop[1] + crop_y1)
                        )

            # -------------------------------------------------
            # Calculate Fish Centre
            # -------------------------------------------------

            if eye_global and tail_global:
                center_x = int(
                    (eye_global[0] + tail_global[0]) / 2
                )
                center_y = int(
                    (eye_global[1] + tail_global[1]) / 2
                )
            else:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

            # -------------------------------------------------
            # Save Coordinates to CSV
            # -------------------------------------------------

            if eye_global:
                eye_x, eye_y = eye_global
            else:
                eye_x, eye_y = -1, -1

            if tail_global:
                tail_x, tail_y = tail_global
            else:
                tail_x, tail_y = -1, -1

            csv_writer.writerow([
                video_name,
                frame_count,
                track_id,
                x1,
                y1,
                x2,
                y2,
                eye_x,
                eye_y,
                tail_x,
                tail_y,
                center_x,
                center_y
            ])

            # -------------------------------------------------
            # Annotated Video
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            if eye_global and tail_global:
                cv2.line(
                    frame,
                    tail_global,
                    eye_global,
                    (0, 255, 255),
                    2
                )

                cv2.circle(
                    frame,
                    eye_global,
                    4,
                    (0, 0, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    tail_global,
                    4,
                    (255, 0, 0),
                    -1
                )

            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # -------------------------------------------------
        # Save Annotated Frame
        # -------------------------------------------------

        out.write(frame)

        cv2.imshow(
            "Fish Coordinate Extraction",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stopped_by_user = True
            break

    # -----------------------------------------------------
    # Cleanup Current Video
    # -----------------------------------------------------

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Finished: {video_name}")
    print(f"Annotated video: {output_video_path}")

    if stopped_by_user:
        print("Processing stopped by user.")
        break

# ---------------------------------------------------------
# 8. Final Cleanup
# ---------------------------------------------------------

data_file.close()
cv2.destroyAllWindows()

print("-----------------------------------------------")
print("Coordinate Extraction Complete!")
print(f"Input folder:  {VIDEO_FOLDER}")
print(f"Video outputs: {VIDEO_DIR}")
print(f"CSV output:    {OUTPUT_DATA_PATH}")
print("-----------------------------------------------")
