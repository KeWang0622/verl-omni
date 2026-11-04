# Reward vs GPU-hours (LTX-2.3 T2AV)

![Reward vs GPU-hours](reward_vs_gpu_hours.png)

- `reward_vs_gpu_hours.png`: left, validation reward vs cumulative GPU-hours (primary); right, the same vs training step.
- `reward_vs_gpu_hours.csv`: one row per plotted point. The flags mark which GPU-hours are estimated and where each reward comes from.
- `reward_vs_gpu_hours.json`: the same points, plus the definitions, threshold crossings and speedups.
- `make_reward_vs_gpuh.py`: regenerates all three from `../metrics.json`:
  `python make_reward_vs_gpuh.py --metrics ../metrics.json --out .`
  Arms run later are added with `--arm NAME=driver.log`.

**Definitions**
- GPU-hours = 8 × cumulative `timing_s/step`, validation steps included.
- Steps without an archived time use the run's median step time, computed separately for validation and non-validation steps. This applies to DiffusionART K=8 after step 26 and K=2 after step 45; the JSON flags these points as `gpu_hours_estimated`.
- Validation rewards at K=8 steps 30–50 and K=2 step 50 were recorded from the run logs at the time and are not in `metrics.json`.
- Threshold crossings are linearly interpolated between consecutive validation points.
- speedup = GPU-hours(baseline reaches threshold) / GPU-hours(ART reaches threshold).

**Pending:** FlowGRPO K=8 and DiffusionART-Flash K=2. Both runs started 2026-11-04 at the PR head.
