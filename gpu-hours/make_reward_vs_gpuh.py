"""Reward vs GPU-hours for verl-project/verl-omni PR #744 (LTX-2.3 T2AV), v2: one hardware, one timing source.

x = 8 × step × (median training-step time on 8×B200 measured on the PR head, validation steps and step 1 excluded),
the same definition for every arm. y = validation reward (mean raw combined reward on the 16 fixed-seed prompts), either
raw or as the gain over the arm's own step-0 (base model) value. Reward-vs-step comes from the October runs
(``metrics.json`` + the validation rewards recorded beyond the archive) for OmniNFT and DiffusionART K=8 / K=2, and from
the current runs' driver logs for FlowGRPO and DiffusionART-Flash.
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
ARMS = {  # validated categorical order (dataviz reference palette); colour follows the arm
    "DiffusionART K=8": dict(color="#2a78d6", marker="o", art=True),
    "OmniNFT (DiffusionNFT)": dict(color="#eb6834", marker="s", art=False),
    "DiffusionART K=2 pairwise": dict(color="#1baf7a", marker="^", art=True),
    "FlowGRPO K=8": dict(color="#eda100", marker="D", art=False),
    "DiffusionART-Flash K=2": dict(color="#e87ba4", marker="v", art=True),
}
ARCHIVE = {  # metrics.json run -> (arm, validation rewards recorded from the run logs beyond the archive)
    "OmniNFT run1": ("OmniNFT (DiffusionNFT)", {}),
    "DiffusionART art_run1": ("DiffusionART K=8", {30: 12.42, 40: 12.83, 50: 12.76}),
    "DiffusionART K=2 pairwise": ("DiffusionART K=2 pairwise", {50: 11.73}),
}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _num(raw: str) -> float:
    return float(re.sub(r"^np\.\w+\((.*)\)$", r"\1", raw.strip().strip("',\"")))


def parse_driver_log(path: Path) -> tuple[dict, dict]:
    """Return ({step: timing_s/step}, {step: val reward}) from a Ray-prefixed verl driver.log."""
    text = path.read_text(errors="replace")
    times, vals = {}, {}
    for line in text.splitlines():
        m = re.search(r"step:(\d+) - (.*)$", line)
        if not m:
            continue
        step = int(m.group(1))
        for key, raw in re.findall(r"(\S+):(\S+?)(?: - |$)", m.group(2)):
            if key == "timing_s/step":
                times[step] = _num(raw)
            elif key == VAL_KEY:
                vals[step] = _num(raw)
    if 0 not in vals:  # "Initial validation metrics: {...}" is pretty-printed over several lines
        m = re.search(r"Initial validation metrics: \{'" + re.escape(VAL_KEY) + r"': \"\s*\n[^\n]*?'(np\.float64\([-0-9.e]+\))", text)
        if m:
            vals[0] = _num(m.group(1))
    return times, vals


def training_step_time(times: dict) -> float:
    """Median step time over non-validation steps, step 1 (warm-up) excluded."""
    picked = [t for s, t in times.items() if s != 1 and s % 5 and s % 10]
    return statistics.median(picked or [t for s, t in times.items() if s != 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True, help="metrics.json of the October runs")
    parser.add_argument("--step-time", action="append", default=[], help="ARM=seconds (B200 median training step time)")
    parser.add_argument("--curve-log", action="append", default=[], help="ARM=driver.log of a current run (val + timing)")
    parser.add_argument("--noise", type=float, default=0.0, help="SD of the base validation reward in the current env")
    parser.add_argument("--relative", action="store_true", help="plot the gain over each arm's own step-0 value")
    parser.add_argument("--thresholds", type=float, nargs=2, default=[11.11, 12.28], help="raw thresholds (October scale)")
    parser.add_argument("--caption", type=Path, default=None, help="text file printed under the figure")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    vals, sources, step_time, timing_source = {}, {}, {}, {}
    data = json.loads(args.metrics.read_text())
    for run, (arm, recorded) in ARCHIVE.items():
        steps = {int(s): v for s, v in data[run].items()}
        vals[arm] = {s: v[VAL_KEY] for s, v in steps.items() if VAL_KEY in v}
        sources[arm] = {s: "October run, archived log" for s in vals[arm]}
        for s, v in recorded.items():
            vals[arm][s] = v
            sources[arm][s] = "October run, recorded from the run log (not archived)"
    for spec in args.curve_log:
        arm, path = spec.rsplit("=", 1)
        times, v = parse_driver_log(Path(path))
        vals[arm] = v
        sources[arm] = {s: "current run (PR head), driver log" for s in v}
        if times:
            step_time[arm] = training_step_time(times)
            timing_source[arm] = f"median of {len(times)} logged steps in the same run"
    for spec in args.step_time:
        arm, sec = spec.rsplit("=", 1)
        step_time[arm] = float(sec)
        timing_source[arm] = "separate 4-step timing run / bench on 8×B200 (PR head)"
    order = [a for a in ARMS if a in vals and a in step_time]
    base_old = vals["OmniNFT (DiffusionNFT)"][0]
    thresholds = [t - base_old for t in args.thresholds] if args.relative else list(args.thresholds)

    curves = {}
    for arm in order:
        base = vals[arm][0]
        current = sources[arm][0].startswith("current")
        curves[arm] = [
            dict(step=s, gpu_hours=GPUS * s * step_time[arm] / 3600, reward=v, value=v - base if args.relative else v,
                 noise=args.noise if current else 0.0, source=sources[arm][s])
            for s, v in sorted(vals[arm].items())
        ]

    def crossing(points, t):
        for a, b in zip(points, points[1:]):
            if a["value"] < t <= b["value"]:
                f = (t - a["value"]) / (b["value"] - a["value"])
                return a["gpu_hours"] + f * (b["gpu_hours"] - a["gpu_hours"])
        return None

    crossings = {a: {str(round(t, 3)): crossing(curves[a], t) for t in thresholds} for a in order}
    speedups = []
    for t in thresholds:
        key = str(round(t, 3))
        for art in (a for a in order if ARMS[a]["art"]):
            for base in (a for a in order if not ARMS[a]["art"]):
                ca, cb = crossings[art][key], crossings[base][key]
                last_b, last_a = curves[base][-1]["gpu_hours"], curves[art][-1]["gpu_hours"]
                if ca is None:
                    value, text = None, f"not reached by {last_a:.0f} GPU-h"
                elif cb is None:
                    value, text = None, f"> {last_b / ca:.2f}x ({base} not reached by {last_b:.0f} GPU-h)"
                else:
                    value, text = cb / ca, f"{cb / ca:.2f}x"
                speedups.append(dict(threshold=t, art=art, baseline=base, speedup=value, text=text))

    with open(args.out / "reward_vs_gpu_hours.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "step", "gpu_hours", "val_reward", "gain_over_step0", "noise_sd", "source"])
        w.writeheader()
        for arm in order:
            base = curves[arm][0]["reward"]
            for p in curves[arm]:
                w.writerow(dict(arm=arm, step=p["step"], gpu_hours=round(p["gpu_hours"], 2), val_reward=round(p["reward"], 4),
                                gain_over_step0=round(p["reward"] - base, 4), noise_sd=p["noise"], source=p["source"]))
    (args.out / "reward_vs_gpu_hours.json").write_text(json.dumps(dict(
        definition=dict(
            gpu_hours="8 x step x median training-step time on 8xB200 at the PR head (validation steps and step 1 excluded)",
            val_reward=f"{VAL_KEY}: mean raw combined reward on the 16 fixed-seed validation prompts",
            y="gain over the arm's own step-0 value" if args.relative else "raw validation reward",
            crossing="first threshold crossing, linearly interpolated between consecutive validation points",
            speedup="GPU-hours(baseline reaches threshold) / GPU-hours(ART reaches threshold)",
        ),
        step_time_s=step_time, timing_source=timing_source, base_noise_sd=args.noise, thresholds=thresholds,
        curves=curves, crossings=crossings, speedups=speedups), indent=1))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7.4), gridspec_kw=dict(width_ratios=[1.65, 1]))
    for a in (ax, ax2):
        a.grid(True, color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    xmax = max(curves[a][-1]["gpu_hours"] for a in order if ARMS[a]["art"] or a.startswith("FlowGRPO")) * 1.08
    for t in thresholds:
        ax.axhline(t, color=INK2, linewidth=1, linestyle=(0, (5, 4)), zorder=1)
        ax.text(xmax * 0.995, t, f"threshold {'+' if args.relative else ''}{t:.2f}", ha="right", va="bottom", fontsize=9, color=INK2)
    for arm in order:
        spec, pts = ARMS[arm], curves[arm]
        shown = [p for p in pts if p["gpu_hours"] <= xmax]
        for panel, xkey, ykey, data_pts in ((ax, "gpu_hours", "value", shown), (ax2, "step", "reward", pts)):
            panel.plot([p[xkey] for p in data_pts], [p[ykey] for p in data_pts], color=spec["color"], linewidth=2, zorder=3)
            panel.errorbar([p[xkey] for p in data_pts], [p[ykey] for p in data_pts], yerr=[p["noise"] for p in data_pts],
                           fmt=spec["marker"], color=spec["color"], markersize=6.5, elinewidth=1, capsize=0, zorder=4,
                           markerfacecolor=SURFACE if data_pts and data_pts[0]["noise"] else spec["color"], markeredgewidth=1.8)
        ax.plot([], [], color=spec["color"], marker=spec["marker"], linewidth=2, label=arm)
        end = shown[-1]
        missed = [t for t in thresholds if spec["art"] and crossings[arm][str(round(t, 3))] is None]
        label = arm.replace("DiffusionART", "ART")
        if len(missed) == len(thresholds):
            label += f"\n(neither threshold reached by {pts[-1]['gpu_hours']:.0f} GPU-h)"
        elif missed:
            label += "".join(f"\n({'+' if args.relative else ''}{t:.2f} not reached by {pts[-1]['gpu_hours']:.0f} GPU-h)" for t in missed)
        if len(shown) < len(pts):
            ax.annotate(label + f"\n(continues to step {pts[-1]['step']}, {pts[-1]['gpu_hours']:.0f} GPU-h)",
                        (end["gpu_hours"], end["value"]), xytext=(0, 10), textcoords="offset points", fontsize=8.5,
                        color=INK, ha="right", va="bottom")
        else:
            ax.annotate(label, (end["gpu_hours"], end["value"]), xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                        color=INK, va="center")
    shown_vals = [p["value"] for a in order for p in curves[a] if p["gpu_hours"] <= xmax]
    ybot, ytop = min(shown_vals + [min(thresholds)]) - 0.45, max(shown_vals + [max(thresholds)]) + 0.5
    ax.set_ylim(ybot, ytop)
    blocks = []
    for t in thresholds:
        key = str(round(t, 3))
        for arm in (a for a in order if not ARMS[a]["art"]):
            c = crossings[arm][key]
            if c is not None and c <= xmax:
                ax.plot(c, t, marker="|", markersize=12, color=ARMS[arm]["color"], zorder=5)
        marks = sorted((crossings[a][key], a) for a in order if ARMS[a]["art"] and crossings[a][key] is not None)
        lines = []
        for c, art in marks:
            if c > xmax:
                continue
            ax.plot([c, c], [ybot, t], color=ARMS[art]["color"], linewidth=1, linestyle=":", zorder=2)
            ax.plot(c, t, marker="|", markersize=12, color=ARMS[art]["color"], zorder=5)
            parts = [f"{sp['text'].replace('x', '×', 1)} vs {sp['baseline'].split(' ')[0]}" for sp in speedups
                     if sp["threshold"] == t and sp["art"] == art and sp["speedup"] is not None]
            if parts:
                lines.append(f"{art.replace('DiffusionART', 'ART')}: " + ", ".join(parts))
        blocks.append(f"Speedup to {'+' if args.relative else ''}{t:.2f}:\n" + ("\n".join(lines) if lines else "no ART arm reached it"))
    ax.set_xlim(0, xmax)
    ax.set_xlabel("GPU-hours (8 × step × median training-step time on 8×B200, PR head)")
    ax.set_ylabel("validation reward gain over step 0" if args.relative else "validation reward")
    ax.set_title("Validation reward vs compute (primary)", loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("validation reward (raw)")
    ax2.set_title("Raw validation reward vs step (secondary)", loc="left", fontsize=12, color=INK)
    caption_h = 0.12 if args.caption else 0.0
    for i, block in enumerate(blocks):
        fig.text(0.04 + 0.31 * i, caption_h + 0.015, block, fontsize=8.6, color=INK, va="bottom", ha="left")
    if args.caption:
        fig.text(0.01, 0.01, Path(args.caption).read_text().strip(), fontsize=8.2, color=INK2, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, caption_h + (0.1 if blocks else 0.02), 1, 1))
    fig.savefig(args.out / "reward_vs_gpu_hours.png", dpi=160)
    print(json.dumps(dict(step_time_s=step_time, speedups=speedups), indent=1))


if __name__ == "__main__":
    main()
