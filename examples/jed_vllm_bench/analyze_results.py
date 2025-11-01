import argparse, json, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.jsonl from bench_client")
    ap.add_argument("--out", default="runs/vllm_bench")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    if not rows:
        print("No rows."); return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir/"raw.csv", index=False)

    # basic stats
    lat = df["latency_s"].values
    summary = {
        "n": int(len(lat)),
        "mean_latency_s": float(np.mean(lat)),
        "p50_s": float(np.percentile(lat, 50)),
        "p90_s": float(np.percentile(lat, 90)),
        "p95_s": float(np.percentile(lat, 95)),
        "p99_s": float(np.percentile(lat, 99)),
        "success_rate": float((df["status"]==200).mean())
    }
    with open(out_dir/"summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(summary)

    # histogram
    plt.figure()
    plt.hist(lat, bins=30)
    plt.xlabel("Latency (s)")
    plt.ylabel("Count")
    plt.title("Latency distribution")
    plt.tight_layout()
    plt.savefig(out_dir/"latency_hist.png")

if __name__ == "__main__":
    main()
