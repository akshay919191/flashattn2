"""Create a backward throughput chart and README table from bench_bwd CSV output."""
import argparse
import csv
import math
import os
import statistics


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "B": int(r["B"]), "H": int(r["H"]), "N": int(r["N"]),
                "D": int(r["D"]), "causal": r["causal"] == "True",
                "custom_ms": float(r["custom_ms"]),
                "custom_tf": float(r["custom_tflops"]),
                "sdpa_ms": float(r["sdpa_ms"]),
                "sdpa_tf": float(r["sdpa_tflops"]),
                "speedup": float(r["speedup"]),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--out", default="docs/figures/backward-throughput.png")
    a = p.parse_args()

    rows = load(a.csv)
    selected = [r for r in rows if r["B"] == a.batch and r["H"] == a.heads]
    dims = sorted({r["D"] for r in selected})
    seqlens = sorted({r["N"] for r in selected})
    if not selected or not dims or not seqlens:
        raise SystemExit(f"no rows for B={a.batch}, H={a.heads} in {a.csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teal, blue = "#13877a", "#2767b0"
    fig, axes = plt.subplots(len(dims), 2, figsize=(10, 4 * len(dims)), squeeze=False,
                             constrained_layout=True)
    width = 0.38
    for i, dim in enumerate(dims):
        for j, (causal, mask) in enumerate(((False, "Dense"), (True, "Causal"))):
            ax = axes[i][j]
            by_n = {r["N"]: r for r in selected
                    if r["D"] == dim and r["causal"] == causal}
            xs = list(range(len(seqlens)))
            custom = [by_n[n]["custom_tf"] if n in by_n else 0.0 for n in seqlens]
            sdpa = [by_n[n]["sdpa_tf"] if n in by_n and not math.isnan(by_n[n]["sdpa_tf"])
                    else 0.0 for n in seqlens]
            ax.bar([x - width / 2 for x in xs], custom, width, label="Custom", color=teal)
            ax.bar([x + width / 2 for x in xs], sdpa, width, label="SDPA", color=blue)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{n:,}" for n in seqlens])
            ax.set_xlabel("Sequence length")
            ax.set_ylabel("TFLOP/s")
            ax.set_title(f"{mask}, D={dim}")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False)
    axes[0][1].legend(frameon=False)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}\n")

    print(f"Execution time (ms), B={a.batch}, H={a.heads}:\n")
    print("| Mask | Head dim | Sequence length | Custom | SDPA |")
    print("|---|---:|---:|---:|---:|")
    for causal, mask in ((False, "Dense"), (True, "Causal")):
        for dim in dims:
            for n in seqlens:
                match = [r for r in selected
                         if r["causal"] == causal and r["D"] == dim and r["N"] == n]
                if match:
                    r = match[0]
                    print(f"| {mask} | {dim} | {n} | {r['custom_ms']:.3f} | {r['sdpa_ms']:.3f} |")

    print(f"\nThe full sweep covers {len(rows)} cases.")
    for causal, mask in ((False, "dense"), (True, "causal")):
        vals = [r["custom_tf"] for r in rows if r["causal"] == causal]
        if vals:
            print(f"Custom {mask} backward throughput: {min(vals):.2f}-{max(vals):.2f} TFLOP/s")
    speedups = [r["speedup"] for r in rows]
    if speedups:
        print(f"Median speedup vs SDPA: {statistics.median(speedups):.2f}x")


if __name__ == "__main__":
    main()
