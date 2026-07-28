#!/usr/bin/env python3
"""
Parse vLLM --enable-logging-iteration-details lines and plot
context vs generation tokens vs iteration index.

Example log line:
  Engine 000: Iteration(7676): 0 context requests, 0 context tokens,
  3 generation requests, 3 generation tokens, iteration elapsed time: 2.97 ms,
  GPU KV cache usage: 0.8%

Usage:
  python3 parse_iteration_logs.py startup_logs/PD1_b4096_colocated_....log
  python3 parse_iteration_logs.py server.log -o iter_tokens.png --engine 0
  python3 parse_iteration_logs.py server.log --max-points 4000 --csv iters.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Match APIServer / EngineCore prefixed lines as well as bare ones.
ITER_RE = re.compile(
    r"Engine\s+(?P<engine>\d+)\s*:\s*Iteration\((?P<iter>\d+)\):\s*"
    r"(?P<ctx_reqs>\d+)\s+context requests,\s*"
    r"(?P<ctx_toks>\d+)\s+context tokens,\s*"
    r"(?P<gen_reqs>\d+)\s+generation requests,\s*"
    r"(?P<gen_toks>\d+)\s+generation tokens,"
    r"(?:.*?iteration elapsed time:\s*(?P<elapsed_ms>[\d.]+)\s*ms)?"
    r"(?:.*?GPU KV cache usage:\s*(?P<kv_pct>[\d.]+)%)?",
    re.IGNORECASE,
)


def parse_log(path: Path, engine: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", errors="replace") as f:
        for line in f:
            if "Iteration(" not in line:
                continue
            m = ITER_RE.search(line)
            if not m:
                continue
            eng = int(m.group("engine"))
            if engine is not None and eng != engine:
                continue
            rows.append(
                {
                    "engine": eng,
                    "iteration": int(m.group("iter")),
                    "context_requests": int(m.group("ctx_reqs")),
                    "context_tokens": int(m.group("ctx_toks")),
                    "generation_requests": int(m.group("gen_reqs")),
                    "generation_tokens": int(m.group("gen_toks")),
                    "elapsed_ms": (
                        float(m.group("elapsed_ms"))
                        if m.group("elapsed_ms") is not None
                        else None
                    ),
                    "kv_cache_pct": (
                        float(m.group("kv_pct")) if m.group("kv_pct") is not None else None
                    ),
                }
            )
    rows.sort(key=lambda r: (r["engine"], r["iteration"]))
    return rows


def downsample(rows: list[dict], max_points: int) -> list[dict]:
    """Bin rows so the plot stays compact for huge iteration counts.

    Within each bin keep mean tokens (and mid iteration index).
    """
    n = len(rows)
    if n <= max_points or max_points < 2:
        return rows

    bin_size = (n + max_points - 1) // max_points
    out: list[dict] = []
    for start in range(0, n, bin_size):
        chunk = rows[start : start + bin_size]
        k = len(chunk)
        out.append(
            {
                "engine": chunk[0]["engine"],
                "iteration": chunk[k // 2]["iteration"],
                "context_requests": sum(r["context_requests"] for r in chunk) / k,
                "context_tokens": sum(r["context_tokens"] for r in chunk) / k,
                "generation_requests": sum(r["generation_requests"] for r in chunk) / k,
                "generation_tokens": sum(r["generation_tokens"] for r in chunk) / k,
                "elapsed_ms": None,
                "kv_cache_pct": None,
                "_binned": k,
            }
        )
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "engine",
        "iteration",
        "context_requests",
        "context_tokens",
        "generation_requests",
        "generation_tokens",
        "elapsed_ms",
        "kv_cache_pct",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def iteration_time_stats(rows: list[dict]) -> dict[str, float] | None:
    """Mean / median / max of per-iteration elapsed_ms (full series, not binned)."""
    xs = [r["elapsed_ms"] for r in rows if r.get("elapsed_ms") is not None]
    if not xs:
        return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mid = n // 2
    median = (
        xs_sorted[mid]
        if n % 2 == 1
        else 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])
    )
    return {
        "mean_ms": sum(xs) / n,
        "median_ms": median,
        "max_ms": xs_sorted[-1],
        "n": float(n),
    }


def plot(
    rows: list[dict],
    out_path: Path,
    *,
    title: str,
    raw_count: int,
    plotted_count: int,
    time_stats: dict[str, float] | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required:  uv pip install matplotlib\n" + str(e)
        ) from e

    xs = [r["iteration"] for r in rows]
    ctx = [r["context_tokens"] for r in rows]
    gen = [r["generation_tokens"] for r in rows]

    # Compact fixed canvas; dense runs stay readable via thin lines + downsample.
    # Dual y-axes: context spikes can be 1000x generation tokens.
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=140)
    ax2 = ax.twinx()

    lw = 1.1 if plotted_count < 3000 else 0.7
    (l1,) = ax.plot(
        xs,
        ctx,
        color="#1f77b4",
        linewidth=lw,
        label="Context tokens",
        alpha=0.95,
    )
    (l2,) = ax2.plot(
        xs,
        gen,
        color="#d62728",
        linewidth=lw,
        label="Generation tokens",
        alpha=0.95,
    )

    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Context tokens", color="#1f77b4")
    ax2.set_ylabel("Generation tokens", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax.set_title(title)
    ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(handles=[l1, l2], loc="upper right", framealpha=0.9)

    if time_stats is not None:
        stats_txt = (
            f"Iteration time (ms):  "
            f"mean={time_stats['mean_ms']:.2f}  "
            f"median={time_stats['median_ms']:.2f}  "
            f"max={time_stats['max_ms']:.2f}"
        )
        # Sit just below the legend in axes coordinates.
        ax.text(
            0.99,
            0.78,
            stats_txt,
            transform=ax.transAxes,
            fontsize=8.5,
            color="#222222",
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.9,
            ),
        )

    note = f"{raw_count} iterations parsed"
    if plotted_count < raw_count:
        note += f" → {plotted_count} points (binned mean)"
    ax.text(
        0.01,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
        va="bottom",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot context/generation tokens from vLLM iteration-detail logs."
    )
    p.add_argument("log_file", type=Path, help="Path to server / startup log")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <log_stem>_iteration_tokens.png)",
    )
    p.add_argument(
        "--engine",
        type=int,
        default=None,
        help="Only keep Engine N (default: all engines; multi-engine plots together)",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Max points to draw; larger runs are binned (default: 5000)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write the full parsed CSV (pre-downsample)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title override",
    )
    args = p.parse_args()

    if not args.log_file.is_file():
        print(f"error: log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    rows = parse_log(args.log_file, engine=args.engine)
    if not rows:
        print(
            f"error: no Iteration(...) lines found in {args.log_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    engines = sorted({r["engine"] for r in rows})
    print(
        f"Parsed {len(rows)} iterations "
        f"(engines={engines}, "
        f"iter {rows[0]['iteration']}→{rows[-1]['iteration']})"
    )

    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote CSV: {args.csv}")

    # If multiple engines without --engine, plot each into separate files.
    if args.engine is None and len(engines) > 1:
        for eng in engines:
            subset = [r for r in rows if r["engine"] == eng]
            plotted = downsample(subset, args.max_points)
            out = args.output
            if out is None:
                out = args.log_file.with_name(
                    f"{args.log_file.stem}_engine{eng:03d}_iteration_tokens.png"
                )
            else:
                out = out.with_name(f"{out.stem}_engine{eng:03d}{out.suffix}")
            title = args.title or (
                f"{args.log_file.name} — Engine {eng:03d}\n"
                f"context tokens & generation tokens vs iteration"
            )
            stats = iteration_time_stats(subset)
            plot(
                plotted,
                out,
                title=title,
                raw_count=len(subset),
                plotted_count=len(plotted),
                time_stats=stats,
            )
            if stats:
                print(
                    f"  iter time ms: mean={stats['mean_ms']:.2f} "
                    f"median={stats['median_ms']:.2f} max={stats['max_ms']:.2f}"
                )
            print(f"Wrote plot: {out}")
        return

    plotted = downsample(rows, args.max_points)
    out = args.output or args.log_file.with_name(
        f"{args.log_file.stem}_iteration_tokens.png"
    )
    eng_label = (
        f"Engine {args.engine:03d}"
        if args.engine is not None
        else (f"Engine {engines[0]:03d}" if len(engines) == 1 else "all engines")
    )
    title = args.title or (
        f"{args.log_file.name} — {eng_label}\n"
        f"context tokens & generation tokens vs iteration"
    )
    stats = iteration_time_stats(rows)
    plot(
        plotted,
        out,
        title=title,
        raw_count=len(rows),
        plotted_count=len(plotted),
        time_stats=stats,
    )
    if stats:
        print(
            f"Iteration time (ms): mean={stats['mean_ms']:.2f} "
            f"median={stats['median_ms']:.2f} max={stats['max_ms']:.2f}"
        )
    print(f"Wrote plot: {out}")


if __name__ == "__main__":
    main()
