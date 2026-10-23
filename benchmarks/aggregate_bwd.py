"""Aggregate repeated bench_bwd CSV sweeps using per-configuration medians."""
import argparse
import csv
import statistics


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True, help="input sweep CSVs")
    p.add_argument("--out", required=True, help="output CSV")
    a = p.parse_args()

    sweeps = [read(path) for path in a.csv]
    if not sweeps or not sweeps[0]:
        raise SystemExit("input CSVs must contain benchmark rows")

    def key(row):
        return (int(row["B"]), int(row["H"]), int(row["N"]), int(row["D"]), row["causal"])

    indexed = [{key(row): row for row in sweep} for sweep in sweeps]
    expected = indexed[0].keys()
    if any(rows.keys() != expected for rows in indexed[1:]):
        raise SystemExit("all input sweeps must contain the same configurations")

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["B", "H", "N", "D", "causal", "custom_ms", "custom_tflops",
                    "sdpa_ms", "sdpa_tflops", "speedup"])
        for config in indexed[0]:
            rows = [sweep[config] for sweep in indexed]
            custom_ms = statistics.median(float(row["custom_ms"]) for row in rows)
            sdpa_ms = statistics.median(float(row["sdpa_ms"]) for row in rows)
            B, H, N, D, causal = config
            flops = 10.0 * B * H * N * N * D * (0.5 if causal == "True" else 1.0)
            w.writerow([B, H, N, D, causal, custom_ms, flops / custom_ms / 1e9,
                        sdpa_ms, flops / sdpa_ms / 1e9, sdpa_ms / custom_ms])

    print(f"Wrote {len(indexed[0])} cases from {len(sweeps)} sweeps to {a.out}")


if __name__ == "__main__":
    main()
