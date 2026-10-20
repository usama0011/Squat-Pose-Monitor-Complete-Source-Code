# Squat Pose Monitor

A Python project for extracting knee angles and counting squat repetitions from videos. It uses YOLOX and RTMPose through RTMLib to detect body landmarks, then calculates the knee angle and tracks the movement from standing to the bottom of a squat and back.

The aim is to explore **human pose estimation and movement analysis**, not to diagnose injury or determine whether someone's exercise technique is clinically correct.

## What it does

- Detects 26 body landmarks using RTMLib `BodyWithFeet` (YOLOX + RTMPose).
- Measures the 2D knee angle using the hip, knee and ankle keypoints.
- Uses median filtering to reduce noisy frame-to-frame angle changes.
- Counts a repetition after the knee moves below a lower threshold and returns to standing.
- Rejects uncertain or multi-person frames instead of randomly switching subjects.
- Saves an annotated video, frame measurements, repetition summaries and a JSON run summary.

## Files

| File | Purpose |
| --- | --- |
| `run_squat.py` | Load a video, run the pose model, draw overlays, save results |
| `analysis.py` | Knee-angle math, landmark checks and the rep counter |
| `tests/test_analysis.py` | Unit tests for angles, counting and uncertain detections |
| `tests/test_video_pipeline.py` | Video export test using a simulated pose model (no weights) |
| `docs/experiment_template.csv` | Table to fill in after checking real-video results |

## Getting started

Use Python 3.10 or 3.11. This example uses Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Put an **unannotated** side-view squat clip inside a local `videos` folder. The folder is excluded from Git:

```powershell
python run_squat.py videos/squats.mp4 --side right --output results --preview
```

If the other leg faces the camera, try `--side left`. The model may download weights the first time it runs; the first run needs internet access. The current setup uses the CPU and RTMLib's `lightweight` models. It can be slow on long videos.

You can experiment with different thresholds:

```powershell
python run_squat.py videos/squats.mp4 --side right --standing-angle 150 --bottom-angle 105
```

The thresholds are starting points for experimentation, **not established clinical cutoffs**. Keep them fixed when comparing clips, and record any changes you make.

## Output

After processing `squats.mp4`, the `results` folder will contain:

- `squats_annotated.mp4` — video with the measured leg and repetition count
- `squats_frames.csv` — per-frame knee angles, counter state, model timing, and tracking status
- `squats_reps.csv` — start/end time, duration, measured range of motion for each rep
- `squats_summary.json` — summary of tracked frames, estimated count, timing, and model settings

The output video keeps the **original video FPS**; `processing_fps` in the JSON is the actual computational throughput. They are not the same measurement.

## How the counting works

1. Wait until the knee is relatively straight (standing position).
2. Detect a move into the bottom position (knee angle decreases).
3. Count one repetition only after the subject stands back up.
4. Require a minimum duration and short confirmation periods to avoid double-counting.
5. If the leg cannot be tracked for too long, discard the unfinished repetition.

Knee angle is calculated from the vectors **knee → hip** and **knee → ankle** using a dot product:

`angle = acos(dot(u, v) / (||u|| * ||v||))`

The angle is computed in the image plane. It depends on camera viewpoint and may be wrong when joints are occluded.

## Testing and current limitations

The repository includes automated tests using **synthetic points, angle traces, and a simulated pose model**. Run them with `python -m unittest discover -s tests -v`.

A passing test means the angle and repetition logic behaved as expected on those examples. It does **not** establish model accuracy on real videos. I still need to compare predictions with hand-counted repetitions and, where possible, reference knee-angle annotations. You can log those results in `docs/experiment_template.csv`.

Only one visible person is supported, ideally filmed from the side with the full leg visible. The project **does not** classify correct vs. incorrect form or make medical recommendations. It uses a pretrained pose estimator; no custom model training is claimed.

## Dependencies and models

This project uses pretrained models provided through [RTMLib](https://github.com/Tau-J/rtmlib). RTMLib and the underlying YOLOX/RTMPose models are separate projects and remain credited to their respective developers. Video data and model weight files are not committed to this repository.
