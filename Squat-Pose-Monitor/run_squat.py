"""Track a single person doing side-view squats in a video.

Usage:
    python run_squat.py videos/squat.mp4 --side right
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from analysis import CountingSettings, LEGS, SquatCounter, leg_measurement


FRAME_COLUMNS = (
    "frame", "time_s", "knee_angle_deg", "smoothed_knee_deg", "phase",
    "rep_count", "tracking_status", "pose_inference_ms"
)
REP_COLUMNS = (
    "rep", "start_s", "end_s", "duration_s", "min_knee_deg",
    "max_knee_deg", "range_of_motion_deg"
)


def load_pose_model():
    # BodyWithFeet uses Halpe26. Other RTMLib classes may use different indexes.
    try:
        from rtmlib import BodyWithFeet
    except ImportError as exc:
        raise RuntimeError("RTMLib is missing. Install: pip install -r requirements.txt") from exc

    return BodyWithFeet(mode="lightweight", backend="onnxruntime",
                        device="cpu", to_openpose=False)


def single_person(keypoints, scores, landmarks, min_confidence):
    """Reject frames with multiple people rather than switching subjects."""
    coords = np.asarray(keypoints)
    conf = np.asarray(scores)
    if coords.size == 0:
        return None, None, "no_person"
    if coords.ndim != 3 or conf.ndim != 2 or len(coords) != len(conf):
        return None, None, "unexpected_model_output"
    if len(coords) != 1:
        return None, None, "multiple_people"
    if coords.shape[1] < 26 or conf.shape[1] < 26:
        raise ValueError("Expected Halpe26 landmarks. Check BodyWithFeet settings.")
    needed = [landmarks["hip"], landmarks["knee"], landmarks["ankle"]]
    if not all(np.isfinite(conf[0, i]) and conf[0, i] >= min_confidence
               and np.all(np.isfinite(coords[0, i, :2])) for i in needed):
        return None, None, "low_confidence"
    return coords[0], conf[0], "tracked"


def annotate(frame, points, indexes, raw_angle, smoothed, counter, status):
    """Draw just the measured leg, keeping the output easy to inspect."""
    view = frame.copy()
    if points is not None:
        leg = (indexes["hip"], indexes["knee"], indexes["ankle"])
        pixels = [tuple(np.rint(points[i, :2]).astype(int)) for i in leg]
        for a, b in zip(pixels, pixels[1:]):
            cv2.line(view, a, b, (0, 210, 255), 3)
        for point in pixels:
            cv2.circle(view, point, 5, (40, 220, 90), -1)

    lines = [
        f"Squats: {counter.count}  |  Phase: {counter.phase}",
        f"Knee angle: {raw_angle:.1f} deg" if raw_angle is not None else "Knee angle: unavailable",
        f"Smoothed: {smoothed:.1f} deg" if smoothed is not None else "Smoothed: unavailable",
        f"Pose: {status}",
    ]
    panel = view.copy()
    cv2.rectangle(panel, (5, 5), (min(640, view.shape[1]-5), 137), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.65, view, 0.35, 0, view)
    for i, line in enumerate(lines):
        cv2.putText(view, line, (18, 32 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def process_video(video_path, output_dir, side, min_confidence, settings, preview):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video metadata is invalid (FPS or image size).")

    annotated_video = output_dir / f"{video_path.stem}_annotated.mp4"
    frame_csv = output_dir / f"{video_path.stem}_frames.csv"
    rep_csv = output_dir / f"{video_path.stem}_reps.csv"
    summary_json = output_dir / f"{video_path.stem}_summary.json"

    writer = cv2.VideoWriter(str(annotated_video),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video. Check MP4 codec support.")

    # Load model once, before timing the actual video-processing loop.
    try:
        model = load_pose_model()
    except Exception:
        writer.release()
        cap.release()
        raise

    counter = SquatCounter(settings)
    indexes = LEGS[side]
    frames = 0
    usable = 0
    inference_ms = []
    status_counts = {}
    start = time.perf_counter()

    try:
        with frame_csv.open("w", newline="", encoding="utf-8") as f:
            log = csv.DictWriter(f, fieldnames=FRAME_COLUMNS)
            log.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frames / fps  # Source-video timeline, not live FPS.
                frames += 1

                tick = time.perf_counter()
                keypoints, scores = model(frame)
                inference_time = 1000 * (time.perf_counter() - tick)
                inference_ms.append(inference_time)

                points, _, status = single_person(keypoints, scores, indexes, min_confidence)
                knee = None
                if points is not None:
                    knee = leg_measurement(points, np.asarray(scores)[0],
                                           side, min_confidence)
                if knee is None:
                    if status == "tracked":
                        status = "invalid_geometry"
                    counter.missing(timestamp)
                    smooth = None
                else:
                    usable += 1
                    smooth, _ = counter.add(knee, timestamp)

                status_counts[status] = status_counts.get(status, 0) + 1
                frame_out = annotate(frame, points, indexes, knee, smooth, counter, status)
                writer.write(frame_out)
                log.writerow({
                    "frame": frames,
                    "time_s": round(timestamp, 4),
                    "knee_angle_deg": round(knee, 2) if knee is not None else "",
                    "smoothed_knee_deg": round(smooth, 2) if smooth is not None else "",
                    "phase": counter.phase,
                    "rep_count": counter.count,
                    "tracking_status": status,
                    "pose_inference_ms": round(inference_time, 2),
                })
                if preview:
                    cv2.imshow("Squat Pose Monitor", frame_out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        writer.release()
        if preview:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    with rep_csv.open("w", newline="", encoding="utf-8") as f:
        log = csv.DictWriter(f, fieldnames=REP_COLUMNS)
        log.writeheader()
        log.writerows(counter.repetitions)

    summary = {
        "input_video": video_path.name,
        "side": side,
        "source_video_fps": round(float(fps), 2),
        "frames_processed": frames,
        "frames_with_valid_knee_angle": usable,
        "completed_reps": counter.count,
        "discarded_cycles": counter.discarded,
        "tracking_statuses": status_counts,
        "processing_seconds": round(elapsed, 3),
        "processing_fps": round(frames / elapsed, 2) if elapsed > 0 else None,
        "average_pose_inference_ms": round(float(np.mean(inference_ms)), 2) if inference_ms else None,
        "model": "RTMLib BodyWithFeet (YOLOX + RTMPose, lightweight mode)",
        "evaluation_status": "No real-video ground-truth accuracy assessed",
        "form_classification": "Not implemented",
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Extract knee angles and count squats from video.")
    parser.add_argument("video", help="Path to an unannotated MP4 squat video")
    parser.add_argument("--side", choices=LEGS, default="right")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--standing-angle", type=float, default=150.0)
    parser.add_argument("--bottom-angle", type=float, default=105.0)
    parser.add_argument("--preview", action="store_true", help="Open OpenCV preview window")
    args = parser.parse_args()

    if not 0 < args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")

    settings = CountingSettings(standing_angle=args.standing_angle,
                                bottom_angle=args.bottom_angle)
    process_video(args.video, args.output, args.side, args.min_confidence,
                  settings, args.preview)


if __name__ == "__main__":
    main()
