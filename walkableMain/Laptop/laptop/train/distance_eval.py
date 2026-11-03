"""Stage 8: evaluate step-length methods for distance; save the best for live mode.

    python -m laptop.train.distance_eval

For every synced session: true pass distances from the video (leg-length
ruler, distance.py), IMU steps from the step CNN, per-step cadence and shin
swing angle. Leave-one-session-out: each method is fitted on the other
sessions' passes and scored on the held-out session (per pass and session
total). The method with the lowest held-out session-total error is refitted
on everything and saved to models/distance.json, with its error as the live
+- band. Figures: models/distance_eval.png.

Steps come from models/folds/<session>.pt when it exists (a step model that
never saw that session); otherwise from the final model, flagged in-sample.
"""

import json

import numpy as np
import pandas as pd

from laptop.config import load_config, resolve
from laptop.train.dataset import Normalizer, load_session
from laptop.train.distance import (METHODS, fit, lengths, px_per_m, sagittal_axis, step_features,
                                   straight_passes)
from laptop.train.events import predict_events
from laptop.train.labels import video_to_imu
from laptop.train.model import load, predict
from laptop.train.train import find_sessions
from laptop.vision.pose import load_landmarks


def session_passes(folder, cfg, final_model):
    s = load_session(folder, cfg, verbose=False)
    lm = load_landmarks(folder / f"landmarks_{cfg['pose']['model']}.parquet")
    scale = px_per_m(lm, cfg)
    to_imu, _ = video_to_imu(folder)
    fold = resolve(cfg["train"]["model_path"]).parent / "folds" / f"{s.name}.pt"
    model, ck = load(fold if fold.exists() else final_model)
    norm = Normalizer(ck["norm"]["mean"], ck["norm"]["std"])
    probs = predict(model, norm(s.x) * s.valid[:, None])
    ev = predict_events(probs, s.rate_hz, cfg["train"]["threshold"], cfg["train"]["refractory_s"])
    gyro = s.x[:, 3:]
    axis = sagittal_axis(gyro[s.valid])
    feats = step_features([s.t_ns[i] for i, _ in ev], [leg for _, leg in ev], s.t_ns, gyro,
                          s.rate_hz, s.meta.get("imu_leg", "R"), axis, cfg)
    passes = []
    for a, b, travel_px in straight_passes(lm, cfg):
        ta, tb = (int(x) for x in to_imu(np.array([a, b])))
        f = feats[(feats.t_ns >= ta) & (feats.t_ns <= tb)]
        if len(f) >= 2:
            passes.append((f, travel_px / scale))
    return {"name": s.name, "speed": s.meta.get("speed"), "scale": scale, "passes": passes,
            "in_sample": not fold.exists(), "axis": axis, "feats": feats}


def evaluate(sessions):
    rows = []
    for k, held in enumerate(sessions):
        train = [p for j, s in enumerate(sessions) if j != k for p in s["passes"]]
        for m in METHODS:
            params = fit(m, train)
            for f, true in held["passes"]:
                est = float(lengths(m, params, f, CFG).sum())
                rows.append({"session": held["name"], "speed": held["speed"], "method": m,
                             "true_m": true, "est_m": est, "steps": len(f)})
    df = pd.DataFrame(rows)
    df["err_pct"] = 100 * (df.est_m - df.true_m) / df.true_m
    return df


def bootstrap_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = [np.mean(np.abs(rng.choice(x, len(x)))) for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def summarize(df):
    out = {}
    for m in METHODS:
        d = df[df.method == m]
        tot = d.groupby("session")[["true_m", "est_m"]].sum()
        sess_err = 100 * (tot.est_m - tot.true_m) / tot.true_m
        lo, hi = bootstrap_ci(d.err_pct)
        out[m] = {
            "pass_mean_abs_pct": float(d.err_pct.abs().mean()), "pass_ci95": [float(lo), float(hi)],
            "session_mean_abs_pct": float(sess_err.abs().mean()),
            "session_err_pct": {k: round(float(v), 1) for k, v in sess_err.items()},
            "by_speed_abs_pct": {k: float(v) for k, v in
                                 d.assign(a=d.err_pct.abs()).groupby("speed").a.mean().items()},
            "bias_pct": float(100 * (tot.est_m.sum() - tot.true_m.sum()) / tot.true_m.sum()),
        }
    return out


def plot(df, sessions, final_params, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"prior": "0.6", "fixed": "tab:blue", "cadence": "tab:orange", "pendulum": "tab:green",
              "walk_ratio_prior": "tab:pink", "walk_ratio": "tab:purple", "swing_speed": "tab:brown"}
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax = axs[0, 0]
    for m in METHODS:
        d = df[df.method == m]
        ax.scatter(d.true_m, d.est_m, s=14, alpha=0.6, color=colors[m], label=m)
    lim = [0, max(df.true_m.max(), df.est_m.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="true pass distance (m, video)", ylabel="estimated (m, IMU)",
           title="held-out passes")
    ax.legend()

    ax = axs[0, 1]
    speeds = ["slow", "normal", "jog"]
    w = 0.8 / len(METHODS)
    for j, m in enumerate(METHODS):
        d = df[df.method == m].assign(a=lambda x: x.err_pct.abs()).groupby("speed").a.mean()
        ax.bar(np.arange(3) + (j - (len(METHODS) - 1) / 2) * w, [d.get(s, np.nan) for s in speeds], w,
               color=colors[m], label=m)
    ax.set(xticks=range(3), xticklabels=speeds, ylabel="mean |error| per pass (%)", title="error by speed")
    ax.legend()

    ax = axs[1, 0]
    for s in sessions:
        f = pd.concat([p for p, _ in s["passes"]]) if s["passes"] else None
        for p, true in s["passes"]:
            ax.scatter(p.freq.mean() * 60, true / len(p), s=14, alpha=0.7,
                       color={"slow": "tab:blue", "normal": "tab:orange", "jog": "tab:red"}.get(s["speed"]))
    c = final_params["cadence"]
    fx = np.linspace(1.0, 2.6, 50)
    ax.plot(fx * 60, c["a"] + c["b"] * fx, "k-", label=f"cadence fit: {c['a']:.2f} + {c['b']:.2f} x f")
    r = final_params["walk_ratio"]["r"]
    r0 = CFG["distance"]["prior_step_m"] / CFG["distance"]["prior_cadence_spm"]
    ax.plot(fx * 60, r * fx * 60, color="tab:purple", label=f"walk ratio fit: {r * 1000:.2f} mm per step/min")
    ax.plot(fx * 60, r0 * fx * 60, color="tab:pink", ls="--", label=f"walk ratio from Apple: {r0 * 1000:.2f}")
    ax.axhline(CFG["distance"]["prior_step_m"], color="0.6", ls="--", label="Apple 0.63 m")
    ax.set(xlabel="cadence (steps/min)", ylabel="step length (m, video / steps)",
           title="step length vs cadence (dots: passes; blue slow, orange normal, red jog)")
    ax.legend()

    ax = axs[1, 1]
    for s in sessions:
        for p, true in s["passes"]:
            ax.scatter(p.pend.mean(), true / len(p), s=14, alpha=0.7,
                       color={"slow": "tab:blue", "normal": "tab:orange", "jog": "tab:red"}.get(s["speed"]))
    k = final_params["pendulum"]["k"]
    x = np.linspace(0.2, 1.0, 20)
    ax.plot(x, k * x, "k-", label=f"fit: {k:.2f} x pendulum")
    ax.plot(x, x, "k:", lw=1, label="pendulum as-is (k = 1)")
    ax.set(xlabel="pendulum step length from shin swing (m)", ylabel="step length (m, video / steps)",
           title="shin swing angle model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close()


CFG = load_config()


def main():
    cfg = CFG
    final_model = resolve(cfg["train"]["model_path"])
    print("sessions:")
    sessions = []
    for f in find_sessions(cfg):
        s = session_passes(f, cfg, final_model)
        tot = sum(m for _, m in s["passes"])
        print(f"  {s['name']}: {s['speed']:7s} {s['scale']:5.0f} px/m, {len(s['passes']):2d} straight passes, "
              f"{tot:5.1f} m, {sum(len(p) for p, _ in s['passes'])} steps"
              + ("  [steps from the FINAL model: in-sample]" if s["in_sample"] else ""))
        sessions.append(s)

    df = evaluate(sessions)
    summary = summarize(df)
    print("\nheld-out results (fit on the other sessions):")
    print(f"  {'method':9s} {'pass |err|':>11s} {'95% CI':>14s} {'session |err|':>14s} {'bias':>7s}   by speed")
    for m, r in summary.items():
        sp = "  ".join(f"{k} {v:.0f}%" for k, v in sorted(r["by_speed_abs_pct"].items()))
        print(f"  {m:9s} {r['pass_mean_abs_pct']:10.1f}% {r['pass_ci95'][0]:6.1f}-{r['pass_ci95'][1]:4.1f}% "
              f"{r['session_mean_abs_pct']:13.1f}% {r['bias_pct']:+6.1f}%   {sp}")
    # Keep the simplest method (fixed) unless another is clearly better on whole sessions
    # AND no worse per pass -- with 8 sessions, a 1-point edge is noise.
    best = "fixed"
    for m in METHODS:
        r, f = summary[m], summary["fixed"]
        if (r["session_mean_abs_pct"] <= f["session_mean_abs_pct"] - 2
                and r["pass_mean_abs_pct"] <= f["pass_mean_abs_pct"]
                and r["session_mean_abs_pct"] < summary[best]["session_mean_abs_pct"]):
            best = m
    all_passes = [p for s in sessions for p in s["passes"]]
    final = {m: fit(m, all_passes) for m in METHODS}
    print(f"\nchosen for live: {best}  (fixed unless another is >= 2 points better per session"
          " and no worse per pass)")
    print("fitted on all sessions: " + "  ".join(f"{m} {v}" for m, v in final.items() if v))

    out = {"method": best, "params": final, "err_frac": summary[best]["session_mean_abs_pct"] / 100,
           "summary": summary, "axis": np.mean([s["axis"] * np.sign(s["axis"][np.argmax(np.abs(s["axis"]))])
                                                for s in sessions], axis=0).tolist(),
           "sessions": [s["name"] for s in sessions],
           "steps_in_sample": any(s["in_sample"] for s in sessions)}
    path = resolve(cfg["distance"]["params_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    df.to_csv(path.with_name("distance_passes.csv"), index=False)
    plot(df, sessions, final, path.with_name("distance_eval.png"))
    print(f"saved {path}\nfigure {path.with_name('distance_eval.png')}")


if __name__ == "__main__":
    main()
