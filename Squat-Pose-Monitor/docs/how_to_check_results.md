# Checking squat-monitor results

Before reporting results on a CV or sending this project to a supervisor, run it on actual footage and compare with independent references.

## Suggested procedure

1. Use short side-view videos with just one person in frame and the full leg visible.
2. Note the camera position and whether it shows the left or right leg.
3. Hand-count the repetitions in each clip without using the program's output.
4. Run the same model and thresholds on all comparison videos.
5. Save the JSON summary and record the measured and hand-counted totals in `experiment_template.csv`.
6. Review a few frames where the knee landmarks fail or the count is wrong.
7. Report absolute repetition error as `abs(predicted reps - reference reps)`.

For angle accuracy, an independently annotated set of reference angles is needed; the calculated angles alone do not demonstrate accuracy. A medical or rehabilitation claim would need substantially stronger independent validation.

## What to check before sharing the GitHub link

- [ ] The setup commands work on your own computer.
- [ ] The output video looks correct.
- [ ] The counter agrees with your manual count on at least one clip, or mismatches are documented.
- [ ] The README only says what has been tested.
- [ ] No personal footage, passport information, API keys, or confidential files are included.
- [ ] You can explain the `joint_angle`, `leg_measurement`, and `SquatCounter.add` functions.
