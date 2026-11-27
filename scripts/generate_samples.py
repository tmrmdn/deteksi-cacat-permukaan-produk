"""Generator citra sampel sintetis: permukaan produk bersih & bercacat.

Permukaan dimodelkan sebagai pelat logam/keramik polos berwarna terang
dengan noise halus. Cacat digambar sebagai bentuk gelap:
- goresan : garis lurus tipis
- noda    : elips/lingkaran gelap
- retak   : garis patah-patah (polyline bergerigi)

Jalankan dari root repo:
    python scripts/generate_samples.py
    python scripts/generate_samples.py --output data/samples

Hasil (800x600, deterministik, seed tetap):
    bersih_01.png   : tanpa cacat (kontrol negatif)
    bersih_02.png   : tanpa cacat, pencahayaan sedikit berbeda
    cacat_01.png    : 2 goresan + 1 noda
    cacat_02.png    : 1 retak + 2 noda + 1 goresan
    cacat_03.png    : 1 retak panjang + 1 goresan + 1 noda besar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 800, 600
BASE_GRAY = 205  # pelat terang; cacat jauh lebih gelap (60-110)


def base_plate(seed: int, brightness: int = BASE_GRAY) -> np.ndarray:
    """Pelat polos + noise Gaussian halus (sigma kecil agar tidak
    terdeteksi sebagai cacat oleh threshold Otsu)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 3, (HEIGHT, WIDTH)).astype(np.float32)
    plate = np.full((HEIGHT, WIDTH), float(brightness), dtype=np.float32)
    plate = np.clip(plate + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(plate, cv2.COLOR_GRAY2BGR)


def scratch(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
            color: int = 70, thickness: int = 3) -> None:
    cv2.line(img, p1, p2, (color, color, color), thickness, cv2.LINE_AA)


def stain(img: np.ndarray, center: tuple[int, int], axes: tuple[int, int],
          color: int = 90, angle: int = 0) -> None:
    cv2.ellipse(img, center, axes, angle, 0, 360,
                (color, color, color), -1, cv2.LINE_AA)


def crack(img: np.ndarray, points: list[tuple[int, int]],
          color: int = 60, thickness: int = 3) -> None:
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=False, color=(color, color, color),
                  thickness=thickness, lineType=cv2.LINE_AA)


def save(img: np.ndarray, path: Path, label: str) -> None:
    cv2.imwrite(str(path), img)
    print(f"{path.name}: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Buat citra sampel sintetis.")
    parser.add_argument("--output", "-o", default="data/samples",
                        help="Folder output (default: data/samples).")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # --- Bersih ---
    save(base_plate(seed=1), out / "bersih_01.png", "bersih, 0 cacat")
    save(base_plate(seed=2, brightness=200), out / "bersih_02.png",
         "bersih, 0 cacat (sedikit lebih gelap)")

    # --- Cacat 01: 2 goresan + 1 noda ---
    img = base_plate(seed=11)
    scratch(img, (120, 150), (420, 170), thickness=3)   # goresan horizontal
    scratch(img, (550, 400), (620, 180), thickness=3)   # goresan diagonal
    stain(img, (300, 420), (55, 38), color=95)          # noda
    save(img, out / "cacat_01.png", "2 goresan + 1 noda")

    # --- Cacat 02: 1 retak + 2 noda + 1 goresan ---
    img = base_plate(seed=12)
    crack(img, [(150, 120), (220, 180), (260, 170), (330, 240),
                (380, 230), (430, 300)], thickness=3)
    stain(img, (580, 150), (42, 30), color=85)
    stain(img, (200, 480), (35, 35), color=100)
    scratch(img, (480, 450), (700, 470), thickness=3)
    save(img, out / "cacat_02.png", "1 retak + 2 noda + 1 goresan")

    # --- Cacat 03: 1 retak panjang + 1 goresan + 1 noda besar ---
    img = base_plate(seed=13)
    crack(img, [(100, 500), (180, 430), (240, 450), (320, 360),
                (400, 380), (470, 290), (540, 300)], thickness=4)
    scratch(img, (150, 100), (500, 90), thickness=3)
    stain(img, (620, 420), (70, 50), color=80, angle=20)
    save(img, out / "cacat_03.png", "1 retak + 1 goresan + 1 noda besar")

    print(f"\nSelesai. {len(list(out.glob('*.png')))} sampel di {out}/")


if __name__ == "__main__":
    main()
