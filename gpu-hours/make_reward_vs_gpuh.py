"""Reward vs GPU-hours figure for verl-project/verl-omni PR #744 (LTX-2.3 T2AV, 8×B200 per arm).

Inputs: ``metrics.json`` from the ``assets-pr744`` branch (archived per-step logs of OmniNFT, DiffusionART K=8 and
K=2) plus the validation rewards recorded from the run logs where the archive stops, and optionally ``driver.log``
files of newer arms (``--arm NAME=PATH``). GPU-hours = 8 × cumulative ``timing_s/step`` (validation steps included),
as in the table posted on the PR. Writes the PNG, a per-point CSV and a JSON with thresholds, crossings and speedups.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VAL_KEY = "val-core/omninft_vggsound/reward/mean@1"
GPUS = 8
# Validated categorical order (dataviz reference palette, light mode); colour follows the arm, never its rank.
ARMS = {
    "DiffusionART K=8": dict(color="#2a78d6", marker="o", art=True),
    "OmniNFT (DiffusionNFT)": dict(color="#eb6834", marker="s", art=False),
    "DiffusionART K=2 pairwise": dict(color="#1baf7a", marker="^", art=True),
    "FlowGRPO K=8": dict(color="#eda100", marker="D", art=False),
    "DiffusionART-Flash K=2": dict(color="#e87ba4", marker="v", art=True),
}
ARCHIVE = {  # metrics.json run name -> (arm, last step, validation rewards recorded from the logs beyond the archive)
    "OmniNFT run1": ("OmniNFT (DiffusionNFT)", 150, {}),
    "DiffusionART art_run1": ("DiffusionART K=8", 50, {30: 12.42, 40: 12.83, 50: 12.76}),
    "DiffusionART K=2 pairwise": ("DiffusionART K=2 pairwise", 50, {50: 11.73}),
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def curve_from_steps(times: dict, vals: dict, last: int, recorded: dict) -> list[dict]:
    """Cumulative GPU-hours per validation step; missing step times use the run's median (val / non-val separately)."""
    non_val = [t for s, t in times.items() if s % 10 and s != 1]
    val_steps = [t for s, t in times.items() if s % 10 == 0]
    med_nv = statistics.median(non_val)
    med_v = statistics.median(val_steps) if val_steps else med_nv
    points = [dict(step=0, gpu_hours=0.0, reward=vals[0], gpu_hours_estimated=False, reward_source="archived log")]
    cum, estimated = 0.0, False
    for step in range(1, last + 1):
        if step in times:
            cum += times[step]
        else:
            cum += med_v if step % 10 == 0 else med_nv
            estimated = True
        if step % 10 == 0:
            if step in vals:
                reward, source = vals[step], "archived log"
            elif step in recorded:
                reward, source = recorded[step], "recorded from the run log (not archived)"
            else:
                continue
            points.append(
                dict(step=step, gpu_hours=GPUS * cum / 3600, reward=reward, gpu_hours_estimated=estimated,
                     reward_source=source)
            )
    return points


def load_archive(path: Path) -> dict:
    data = json.loads(path.read_text())
    curves = {}
    for run, (arm, last, recorded) in ARCHIVE.items():
        steps = {int(s): v for s, v in data[run].items()}
        times = {s: v["timing_s/step"] for s, v in steps.items() if "timing_s/step" in v}
        vals = {s: v[VAL_KEY] for s, v in steps.items() if VAL_KEY in v}
        curves[arm] = curve_from_steps(times, vals, last, recorded)
    return curves


def load_driver_log(path: Path) -> list[dict]:
    """Per-step times and validation rewards from a verl driver.log (Ray-prefixed ``step:N - key:value`` lines)."""
    times, vals = {}, {}
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"step:(\d+) - (.*)$", line)
        if not m:
            continue
        step = int(m.group(1))
        for key, raw in re.findall(r"(\S+):(\S+?)(?: - |$)", m.group(2)):
            raw = re.sub(r"^np\.\w+\((.*)\)$", r"\1", raw)
            if key == "timing_s/step":
                times[step] = float(raw)
            elif key == VAL_KEY:
                vals[step] = float(raw)
    last = max([s for s in vals if s % 10 == 0] or [0])
    return curve_from_steps(times, vals, last, {}) if times else [
        dict(step=0, gpu_hours=0.0, reward=vals[0], gpu_hours_estimated=False, reward_source="driver log")
    ]


def first_crossing(points: list[dict], threshold: float):
    """GPU-hours where the curve first reaches ``threshold``, linearly interpolated between validation points."""
    for a, b in zip(points, points[1:]):
        if a["reward"] < threshold <= b["reward"]:
            frac = (threshold - a["reward"]) / (b["reward"] - a["reward"])
            return dict(gpu_hours=a["gpu_hours"] + frac * (b["gpu_hours"] - a["gpu_hours"]), step_between=[a["step"], b["step"]],
                        uses_estimated=b["gpu_hours_estimated"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[], help="NAME=driver.log for arms run after the archive")
    parser.add_argument("--thresholds", type=float, nargs=2, default=[11.11, 12.28])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    curves = load_archive(args.metrics)
    for spec in args.arm:
        name, path = spec.split("=", 1)
        curves[name] = load_driver_log(Path(path))
    order = [arm for arm in ARMS if arm in curves]

    crossings = {arm: {str(t): first_crossing(curves[arm], t) for t in args.thresholds} for arm in order}
    speedups = []
    for t in args.thresholds:
        for art in (a for a in order if ARMS[a]["art"]):
            for base in (a for a in order if not ARMS[a]["art"]):
                c_art, c_base = crossings[art][str(t)], crossings[base][str(t)]
                last_base = curves[base][-1]["gpu_hours"]
                if c_art is None:
                    value, text = None, f"{art} does not reach {t} (last point {curves[art][-1]['gpu_hours']:.1f} GPU-h)"
                elif c_base is None:
                    value, text = None, f"> {last_base / c_art['gpu_hours']:.2f}x ({base} not reached by {last_base:.1f} GPU-h)"
                else:
                    value = c_base["gpu_hours"] / c_art["gpu_hours"]
                    text = f"{value:.2f}x"
                speedups.append(dict(threshold=t, art=art, baseline=base, speedup=value, text=text))

    with open(args.out / "reward_vs_gpu_hours.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "step", "gpu_hours", "val_reward", "gpu_hours_estimated", "reward_source"])
        writer.writeheader()
        for arm in order:
            for p in curves[arm]:
                writer.writerow(dict(arm=arm, step=p["step"], gpu_hours=round(p["gpu_hours"], 2), val_reward=round(p["reward"], 4),
                                     gpu_hours_estimated=p["gpu_hours_estimated"], reward_source=p["reward_source"]))
    (args.out / "reward_vs_gpu_hours.json").write_text(json.dumps(dict(
        definition=dict(
            gpu_hours="8 x cumulative timing_s/step (validation steps included); steps without an archived time use the "
                      "run's median step time (validation and non-validation steps separately)",
            val_reward=f"{VAL_KEY}: mean raw combined reward on the fixed-seed validation prompts",
            crossing="first threshold crossing, linearly interpolated between consecutive validation points",
            speedup="GPU-hours(baseline reaches threshold) / GPU-hours(ART reaches threshold); '> Nx' when the baseline "
                    "does not reach it within its run (N from its last point)",
        ),
        thresholds=args.thresholds, curves={a: curves[a] for a in order}, crossings=crossings, speedups=speedups), indent=1))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1.65, 1]))
    xmax = 160.0
    for a in (ax, ax2):
        a.grid(True, color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    for t in args.thresholds:
        for a in (ax, ax2):
            a.axhline(t, color=INK2, linewidth=1, linestyle=(0, (5, 4)), zorder=1)
        ax.text(xmax * 0.995, t + 0.05, f"threshold {t:.2f}", ha="right", va="bottom", fontsize=9, color=INK2)

    for arm in order:
        spec, pts = ARMS[arm], curves[arm]
        for panel, xkey in ((ax, "gpu_hours"), (ax2, "step")):
            for a_pt, b_pt in zip(pts, pts[1:]):
                panel.plot([a_pt[xkey], b_pt[xkey]], [a_pt["reward"], b_pt["reward"]], color=spec["color"], linewidth=2,
                           linestyle=(0, (4, 3)) if b_pt["gpu_hours_estimated"] else "-", zorder=3)
            for p in pts:
                hollow = p["gpu_hours_estimated"] or p["reward_source"].startswith("recorded")
                panel.plot(p[xkey], p["reward"], marker=spec["marker"], markersize=6.5, color=spec["color"],
                           markerfacecolor=SURFACE if hollow else spec["color"], markeredgewidth=1.8, zorder=4,
                           linestyle="none")
        inside = [p for p in pts if p["gpu_hours"] <= xmax]
        end = inside[-1]
        missed = [t for t in args.thresholds if spec["art"] and crossings[arm][str(t)] is None]
        label = arm + "".join(f"\n({t:.2f} not reached by {pts[-1]['gpu_hours']:.0f} GPU-h)" for t in missed)
        if len(inside) < len(pts):  # runs past the right edge: label above the line, ending at the edge
            beyond = pts[-1]
            label += f"\n(continues to {beyond['reward']:.2f} at {beyond['gpu_hours']:.0f} GPU-h, step {beyond['step']})"
            ax.annotate(label, (end["gpu_hours"], end["reward"]), xytext=(0, 12), textcoords="offset points",
                        fontsize=9, color=INK, ha="right", va="bottom")
        else:
            ax.annotate(label, (end["gpu_hours"], end["reward"]), xytext=(7, 0), textcoords="offset points",
                        fontsize=9, color=INK, va="center")
        ax.plot([], [], color=spec["color"], marker=spec["marker"], linewidth=2, markersize=6.5, label=arm)

    for t in args.thresholds:
        marks = sorted(
            ((crossings[sp["art"]][str(t)]["gpu_hours"], i, sp) for i, sp in enumerate(speedups)
             if sp["threshold"] == t and crossings[sp["art"]][str(t)] is not None and sp["baseline"].startswith("OmniNFT")),
            key=lambda m: m[0],
        )
        for j, (x, _, sp) in enumerate(marks):
            ax.plot([x, x], [8.9, t], color=ARMS[sp["art"]]["color"], linewidth=1, linestyle=":", zorder=2)
            ax.plot(x, t, marker="|", markersize=12, color=ARMS[sp["art"]]["color"], zorder=5)
            crowded = j + 1 < len(marks) and marks[j + 1][0] - x < 12
            ax.annotate(sp["text"].replace("x", "×"), (x, 8.9), xytext=(-4 if crowded else 4, 5), textcoords="offset points",
                        fontsize=9.5, color=INK, ha="right" if crowded else "left", va="bottom")
    for arm in (a for a in order if not ARMS[a]["art"]):
        for t in args.thresholds:
            c = crossings[arm][str(t)]
            if c is not None:
                ax.plot(c["gpu_hours"], t, marker="|", markersize=12, color=ARMS[arm]["color"], zorder=5)

    ax.set_xlim(0, xmax)
    ax.set_ylim(8.9, 14.3)
    ax.set_xlabel("cumulative GPU-hours (8 × cumulative step time, B200)")
    ax.set_ylabel("validation reward (mean raw combined reward)")
    ax.set_title("Validation reward vs compute (primary)", loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_xlabel("training step")
    ax2.set_title("Validation reward vs step (secondary)", loc="left", fontsize=12, color=INK)
    ax2.set_ylim(8.9, max(p["reward"] for a in order for p in curves[a]) + 0.3)
    caption = (
        "LTX-2.3 T2AV, identical prompts, seeds, rewards, routing, LoRA r32, lr and sampler; one 8×B200 node per arm; validation every 10 steps "
        "(step 0 = base model, 9.10 for all arms). Solid / filled: step times and validation rewards from the archived logs. Dashed / hollow: "
        "GPU-hours estimated from the run's median step time (DiffusionART K=8 after step 26, K=2 after step 45) and validation rewards recorded "
        "from the run logs but not archived (K=8 steps 30–50, K=2 step 50). Crossings are linear interpolations between validation points; "
        "dotted drop lines mark DiffusionART's first crossing, labelled with the speedup vs OmniNFT = GPU-hours(OmniNFT reaches threshold) / "
        "GPU-hours(ART reaches threshold); orange ticks mark OmniNFT's crossings. The left panel is cut at 160 GPU-h; the right panel shows OmniNFT "
        "to step 150. FlowGRPO and DiffusionART-Flash curves: runs in progress."
    )
    fig.text(0.01, 0.01, caption, fontsize=8.2, color=INK2, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(args.out / "reward_vs_gpu_hours.png", dpi=160)
    print(json.dumps(speedups, indent=1))


if __name__ == "__main__":
    main()
