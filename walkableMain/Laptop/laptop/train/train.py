"""Stage 6: train + evaluate the IMU heel-strike CNN on synced sessions.

    python -m laptop.train.train                  # all good sessions in data/recordings
    python -m laptop.train.train --sessions data/recordings/A data/recordings/B
    python -m laptop.train.train --relabel        # redo vision labels
    python -m laptop.train.train --iterations 300 # quick run

1. Labels: hand truth.csv if the session has one, else the vision detector,
   moved onto the IMU clock with the stomp correction (labels.py).
2. Leave-one-session-out: for each session, train on all the others and
   score the held-out one (precision / recall / F1 / timing, per leg and
   leg-blind), next to a no-learning baseline (accel-magnitude peaks).
3. Final model on all sessions -> models/step_cnn.pt, plus
   models/step_cnn_report.json and models/step_cnn_eval.png.
Scores only count where labels can be trusted (between the stomps, heels
visible to the camera). Labels from vision are themselves imperfect: the
scores measure agreement with the camera, not with the truth.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from laptop.config import load_config, resolve
from laptop.train.dataset import Normalizer, load_session, sample_windows
from laptop.train.events import pick_peaks, predict_events, score
from laptop.train.model import StepCNN, predict, save


def find_sessions(cfg):
    tc = cfg["train"]
    out = []
    for f in sorted(resolve(tc["sessions_dir"]).glob("*/")):
        meta_p = f / "meta.json"
        if not (meta_p.exists() and (f / "imu.parquet").exists() and (f / "video.mp4").exists()):
            continue
        notes = json.load(open(meta_p)).get("notes", "").lower()
        if any(w in notes for w in tc["skip_notes"]):
            print(f"  skip {f.name}: notes say '{notes}'")
            continue
        if tc["require_stomp_sync"] and not (f / "sync_correction.json").exists():
            print(f"  skip {f.name}: no stomp sync (run tests/hw_stage4_sync.py --check {f})")
            continue
        out.append(f)
    return out


def train_model(sessions, cfg, iterations=None, log_every=200, seed=None):
    tc = cfg["train"]
    seed = tc["seed"] if seed is None else seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = tc["device"]
    norm = Normalizer.fit(sessions)
    rate = float(np.median([s.rate_hz for s in sessions]))
    length = int(round(tc["window_s"] * rate))
    model = StepCNN(ch=tc["channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    pos_w = torch.tensor(tc["pos_weight"], device=device)
    iterations = iterations or tc["iterations"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations)
    t0 = time.time()
    model.train()
    for it in range(1, iterations + 1):
        X, Y, M = sample_windows(sessions, norm, tc["batch"], length, tc["min_labeled_frac"], rng,
                                 tc["augment_gain"], tc["augment_noise"])
        X, Y, M = (torch.as_tensor(a, device=device) for a in (X, Y, M))
        logits = model(X)
        # Soft targets: positives weighted up; unlabeled samples don't count.
        bce = nn.functional.binary_cross_entropy_with_logits(logits, Y, reduction="none")
        w = 1 + (pos_w - 1) * Y
        loss = (bce * w * M[:, None]).sum() / (M.sum() * Y.shape[1] + 1e-6)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and it % log_every == 0:
            print(f"    iter {it:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f} s)")
    return model, norm


def baseline_events(s, cfg):
    """No learning: peaks of accel magnitude, leg-blind (reported as leg L)."""
    dyn = np.linalg.norm(s.x[:, :3], axis=1)
    dyn = (dyn - np.median(dyn[s.valid])) / (np.std(dyn[s.valid]) + 1e-6)
    peaks = pick_peaks(dyn, s.rate_hz, threshold=2.0, refractory_s=cfg["train"]["refractory_s"] / 2)
    return [(i, "L") for i in peaks]


def evaluate(model, norm, s, cfg):
    tc = cfg["train"]
    probs = predict(model, norm(s.x) * s.valid[:, None], tc["device"])
    pred = predict_events(probs, s.rate_hz, tc["threshold"], tc["refractory_s"])
    r = score(pred, s.events, s.labeled, s.rate_hz, tc["match_tol_s"])
    b = score(baseline_events(s, cfg), [(i, "L") for i, _ in s.events], s.labeled, s.rate_hz,
              tc["match_tol_s"])
    return r, b, probs, pred


def fmt(r):
    t = "--" if r["timing_median_abs_ms"] is None else f"{r['timing_median_abs_ms']:.0f}"
    return (f"P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f}  "
            f"{r['matched']}/{r['truth']} matched, {r['predicted']} predicted  |t| {t} ms")


def plot_eval(results, path, seconds=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(results), 1, figsize=(16, 3.2 * len(results)), squeeze=False)
    for ax, (s, probs, pred) in zip(axs[:, 0], results):
        lab = np.flatnonzero(s.labeled)
        a = lab[0] if len(lab) else 0
        b = min(len(s.t_ns), a + int(seconds * s.rate_hz))
        tt = (np.arange(a, b) - a) / s.rate_hz
        ax.plot(tt, np.linalg.norm(s.x[a:b, :3], axis=1) / 9.81 / 4, color="0.7", lw=0.8,
                label="|accel| / 4 g")
        for k, (leg, c) in enumerate((("L", "tab:blue"), ("R", "tab:orange"))):
            ax.plot(tt, probs[a:b, k], color=c, lw=1.2, label=f"P(strike {leg})")
            for i, l in s.events:
                if l == leg and a <= i < b:
                    ax.axvline((i - a) / s.rate_hz, color=c, ls="--", lw=0.8)
            for i, l in pred:
                if l == leg and a <= i < b:
                    ax.plot((i - a) / s.rate_hz, probs[i, k], "v", color=c, ms=8)
        ax.set_title(f"held-out {s.name}: dashed = labels, triangles = CNN steps")
        ax.set_xlabel("s")
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close()


def main():
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", nargs="*", help="session folders (default: all good ones)")
    p.add_argument("--relabel", action="store_true")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--no-final", action="store_true", help="skip the final all-session model")
    args = p.parse_args()

    print("sessions:")
    folders = [Path(s) for s in args.sessions] if args.sessions else find_sessions(cfg)
    sessions = []
    for f in folders:
        s = load_session(f, cfg, args.relabel)
        src = "hand" if (f / "truth.csv").exists() else "vision"
        print(f"  {s.name}: {len(s.t_ns) / s.rate_hz:.0f} s IMU, {len(s.events)} {src} labels, "
              f"{s.labeled.mean():.0%} of samples labeled, speed {s.meta.get('speed')}")
        sessions.append(s)
    if not sessions:
        raise SystemExit("no usable sessions (record with python -m laptop.sync.record)")

    report = {"sessions": [s.name for s in sessions], "folds": {}}
    plots = []
    if len(sessions) >= 2:
        print("\nleave-one-session-out:")
        for k, held in enumerate(sessions):
            rest = [s for j, s in enumerate(sessions) if j != k]
            print(f"  fold {k + 1}/{len(sessions)}: train on {len(rest)}, test {held.name}")
            model, norm = train_model(rest, cfg, args.iterations, log_every=0, seed=cfg["train"]["seed"] + k)
            # Kept so Stage 8 can measure distance on this session with a model that never saw it.
            fold_dir = resolve(cfg["train"]["model_path"]).parent / "folds"
            fold_dir.mkdir(parents=True, exist_ok=True)
            save(fold_dir / f"{held.name}.pt", model, norm, cfg["train"],
                 {"sessions": [s.name for s in rest], "rate_hz": held.rate_hz})
            r, b, probs, pred = evaluate(model, norm, held, cfg)
            print(f"    CNN      {fmt(r)}  (leg-blind matched {r['leg_blind_matched']})")
            print(f"    baseline {fmt(b)}  (leg-blind accel peaks)")
            report["folds"][held.name] = {"cnn": r, "baseline": b, "speed": held.meta.get("speed")}
            plots.append((held, probs, pred))
        f1 = [v["cnn"]["f1"] for v in report["folds"].values()]
        bf1 = [v["baseline"]["f1"] for v in report["folds"].values()]
        report["mean_f1"], report["baseline_mean_f1"] = float(np.mean(f1)), float(np.mean(bf1))
        print(f"\n  mean held-out F1: CNN {np.mean(f1):.2f}  vs baseline {np.mean(bf1):.2f}")
    else:
        print("\nonly 1 session: no held-out evaluation possible (record more)")

    out = resolve(cfg["train"]["model_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_final:
        print(f"\nfinal model on all {len(sessions)} session(s) ...")
        model, norm = train_model(sessions, cfg, args.iterations)
        save(out, model, norm, cfg["train"], {"sessions": report["sessions"],
                                              "rate_hz": float(np.median([s.rate_hz for s in sessions]))})
        print(f"saved {out}")
    with open(out.with_name(out.stem + "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    if plots:
        plot_eval(plots, out.with_name(out.stem + "_eval.png"))
        print(f"report {out.with_name(out.stem + '_report.json')}\nplot   {out.with_name(out.stem + '_eval.png')}")


if __name__ == "__main__":
    main()
