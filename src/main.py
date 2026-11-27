"""CLI demo: deteksi cacat permukaan produk dari satu citra / folder.

Contoh:
    python -m src.main --input data/samples/cacat_01.png --output results/
    python -m src.main --input data/samples/ --output results/
    python -m src.main --input data/samples/cacat_01.png --output results/ --min-area 80
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Agar `python -m src.main` jalan dari root repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detect import process_file  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi cacat permukaan produk menggunakan segmentasi citra."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="File citra (.png/.jpg) atau folder berisi citra.",
    )
    parser.add_argument(
        "--output", "-o", default="results",
        help="Folder output (default: results).",
    )
    parser.add_argument(
        "--min-area", type=float, default=40,
        help="Luas kontur minimum (px) agar dianggap cacat (default: 40).",
    )
    parser.add_argument(
        "--px-per-mm", type=float, default=None,
        help="Opsional: jumlah piksel per mm, untuk mengonversi luas ke mm2.",
    )
    return parser.parse_args(argv)


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_inputs(input_path)
    if not files:
        print(f"Tidak ada citra di: {input_path}")
        return 1

    rows = []
    for file in files:
        result, annotated_path, json_path = process_file(
            file, output_dir, min_area=args.min_area, px_per_mm=args.px_per_mm
        )
        from collections import Counter

        counter = Counter(d.jenis for d in result.cacat)
        jenis = ", ".join(f"{k} x{v}" for k, v in counter.items()) or "-"

        print(
            f"{result.file}: {result.jumlah_cacat} cacat "
            f"({jenis}) | luas total {result.luas_total_cacat_px:.0f} px "
            f"({result.persentase_area_cacat:.2f}% area) "
            f"| Otsu={result.threshold_otsu}"
        )
        print(f"  -> {annotated_path}")
        print(f"  -> {json_path}")
        rows.append(
            {
                "file": result.file,
                "jumlah_cacat": result.jumlah_cacat,
                "luas_total_cacat_px": result.luas_total_cacat_px,
                "persentase_area_cacat": result.persentase_area_cacat,
                "threshold_otsu": result.threshold_otsu,
            }
        )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ringkasan CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
