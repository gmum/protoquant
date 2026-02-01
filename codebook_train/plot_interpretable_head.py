#!/usr/bin/env python3
"""Plot Interpretable Head Top-1 Validation Accuracy vs. Codebook Size."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def parse_interpretable_section(text: str):
    """Return list of (accuracy, size) from the Interpretable Head section."""
    marker = "(Interpretable Head)"
    start = text.find(marker)
    if start == -1:
        raise ValueError("Could not find '(Interpretable Head)' section.")

    # Stop at next section or end of file
    rest = text[start + len(marker) :]
    end = rest.find("(Non Interpretable Head)")
    if end != -1:
        rest = rest[:end]

    # Collect numeric tokens (floats or ints)
    tokens = []
    for line in rest.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tokens.append(float(line))
        except ValueError:
            # skip non-numeric lines
            pass

    # Expect pairs: accuracy, size
    pairs = []
    i = 0
    while i + 1 < len(tokens):
        acc = tokens[i]
        size = int(tokens[i + 1])
        pairs.append((acc, size))
        i += 2

    if not pairs:
        raise ValueError("No numeric pairs found in Interpretable Head section.")

    # Sort by codebook size
    pairs.sort(key=lambda x: x[1])
    return pairs


def plot_series(
    interpretable,
    non_interpretable,
    out_path: Path,
    title: str | None,
    show_error: bool,
):
    plt.figure(figsize=(6.8, 4.0))

    all_sizes: list[int] = []
    if interpretable:
        sizes_i = [p[0] for p in interpretable]
        means_i = [p[1] for p in interpretable]
        stds_i = [p[2] for p in interpretable] if show_error else None
        all_sizes.extend(sizes_i)
        plt.errorbar(
            sizes_i,
            means_i,
            yerr=stds_i,
            marker="o",
            linewidth=1.6,
            color="#1f77b4",
            capsize=3,
            label="Interpretable head",
        )

    if non_interpretable:
        sizes_n = [p[0] for p in non_interpretable]
        means_n = [p[1] for p in non_interpretable]
        stds_n = [p[2] for p in non_interpretable] if show_error else None
        all_sizes.extend(sizes_n)
        plt.errorbar(
            sizes_n,
            means_n,
            yerr=stds_n,
            marker="s",
            linewidth=1.6,
            color="#ff7f0e",
            capsize=3,
            label="Pre-trained head",
        )

    if not all_sizes:
        raise ValueError("No data provided to plot.")

    unique_sizes = sorted(set(all_sizes))
    plt.xscale("log", base=2)
    plt.xticks(unique_sizes, [str(s) for s in unique_sizes])
    plt.xlabel("Codebook size")
    plt.ylabel("Top-1 Validation Accuracy (%)")
    if title:
        plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".png"), dpi=300)
    plt.savefig(out_path.with_suffix(".pdf"))


def _select_lr(values: Iterable[float], lr: float | None) -> float:
    unique = sorted(set(values))
    if lr is not None:
        if lr not in unique:
            raise ValueError(
                f"Requested lr={lr} not found in CSVs. Available: {unique}"
            )
        return lr
    if len(unique) == 1:
        return unique[0]
    raise ValueError(
        f"Multiple learning rates found: {unique}. Use --lr to select one."
    )


def parse_csv_pairs(csv_paths: list[Path], lr: float | None, error_mode: str):
    rows: list[tuple[float, int, float]] = []  # (lr, size, acc)
    for path in csv_paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    acc = float(row["Validation Top1 Accuracy"])
                    size = int(float(row["codebook.num_entries"]))
                    lr_val = float(row["base_optimizer.lr"])
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Malformed row in {path}: {row}") from exc
                rows.append((lr_val, size, acc))

    if not rows:
        raise ValueError("No rows found in CSVs.")

    selected_lr = _select_lr([r[0] for r in rows], lr)

    grouped: dict[int, list[float]] = {}
    for lr_val, size, acc in rows:
        if lr_val != selected_lr:
            continue
        grouped.setdefault(size, []).append(acc)

    pairs = []
    for size, accs in grouped.items():
        mean = float(statistics.mean(accs))
        if error_mode == "std":
            err = float(statistics.stdev(accs)) if len(accs) > 1 else 0.0
        elif error_mode == "sem":
            err = (
                float(statistics.stdev(accs)) / (len(accs) ** 0.5)
                if len(accs) > 1
                else 0.0
            )
        elif error_mode == "ci95":
            err = (
                1.96 * float(statistics.stdev(accs)) / (len(accs) ** 0.5)
                if len(accs) > 1
                else 0.0
            )
        elif error_mode == "none":
            err = 0.0
        else:
            raise ValueError(f"Unknown error_mode: {error_mode}")
        pairs.append((size, mean, err))

    pairs.sort(key=lambda x: x[0])
    return pairs


def parse_pairs_arg(pairs_text: str):
    """Parse pairs from a string like 'acc,size; acc,size; ...'."""
    pairs = []
    for chunk in pairs_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid pair '{chunk}'. Use 'acc,size'.")
        acc = float(parts[0])
        size = int(parts[1])
        pairs.append((acc, size))
    if not pairs:
        raise ValueError("No pairs provided.")
    pairs.sort(key=lambda x: x[1])
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Plot Interpretable Head accuracy vs. codebook size."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to results_comparison_entries.txt",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="Interpretable pairs as 'acc,size; acc,size; ...' (legacy).",
    )
    parser.add_argument(
        "--non-pairs",
        type=str,
        default=None,
        help="Non-interpretable pairs as 'acc,size; acc,size; ...' (legacy).",
    )
    parser.add_argument(
        "--interpretable-csv",
        type=Path,
        nargs="+",
        default=None,
        help="CSV(s) for interpretable head results.",
    )
    parser.add_argument(
        "--non-interpretable-csv",
        type=Path,
        nargs="+",
        default=None,
        help="CSV(s) for pre-trained (non-interpretable) head results.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate to select when CSV contains multiple values.",
    )
    parser.add_argument(
        "--error",
        type=str,
        default="std",
        choices=["std", "sem", "ci95", "none"],
        help="Error bars: std, sem, ci95, or none. Default: std.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("interpretable_head_plot"),
        help="Output path without extension (png/pdf will be saved).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Interpretable Head",
        help="Optional plot title (use '' for no title).",
    )
    args = parser.parse_args()

    interpretable = None
    non_interpretable = None

    if args.interpretable_csv:
        interpretable = parse_csv_pairs(args.interpretable_csv, args.lr, args.error)
    if args.non_interpretable_csv:
        non_interpretable = parse_csv_pairs(
            args.non_interpretable_csv, args.lr, args.error
        )

    if args.pairs:
        pairs = parse_pairs_arg(args.pairs)
        interpretable = [(size, acc, 0.0) for acc, size in pairs]
    if args.non_pairs:
        pairs = parse_pairs_arg(args.non_pairs)
        non_interpretable = [(size, acc, 0.0) for acc, size in pairs]

    if interpretable is None and non_interpretable is None:
        if args.input is None:
            args.input = Path("results_comparison_entries.txt")
        text = args.input.read_text()
        pairs = parse_interpretable_section(text)
        interpretable = [(size, acc, 0.0) for acc, size in pairs]

    title = args.title if args.title else None
    show_error = args.error != "none"
    plot_series(interpretable, non_interpretable, args.output, title, show_error)


if __name__ == "__main__":
    main()
