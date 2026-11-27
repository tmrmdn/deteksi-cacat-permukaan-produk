"""
Deteksi cacat permukaan produk menggunakan segmentasi citra.

Pipeline:
    1. Baca citra -> grayscale
    2. Gaussian blur (mengurangi noise)
    3. Threshold Otsu (otomatis, citra dibalik: cacat yang gelap
       menjadi foreground putih)
    4. Morfologi opening (buang noise kecil) + closing (tutup lubang)
    5. Deteksi kontur -> filter berdasarkan luas
    6. Klasifikasi cacat dari bentuk kontur:
       goresan / retak / noda (heuristik bentuk, lihat classify_defect)

Judul tugas: Deteksi Cacat Permukaan Produk Menggunakan Segmentasi Citra
Mata kuliah: Pengolahan Citra Digital (22TIF0262), Teknik Informatika S1
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# Warna anotasi (BGR, OpenCV) per jenis cacat
COLORS = {
    "goresan": (0, 255, 255),   # kuning
    "retak": (255, 0, 255),     # magenta
    "noda": (0, 0, 255),        # merah
}

# Luas kontur minimum (px) agar dianggap cacat, bukan noise.
MIN_AREA = 40


@dataclass
class Defect:
    """Satu cacat yang terdeteksi."""

    id: int
    jenis: str
    luas_px: float
    luas_mm2: float | None
    bbox: list[int]  # [x, y, w, h]
    aspect_ratio: float
    solidity: float
    extent: float
    linearitas: float


@dataclass
class DetectionResult:
    """Ringkasan hasil deteksi untuk satu citra."""

    file: str
    ukuran_citra: list[int]  # [lebar, tinggi]
    jumlah_cacat: int
    luas_total_cacat_px: float
    persentase_area_cacat: float
    cacat: list[Defect]
    threshold_otsu: float
    threshold_digunakan: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def classify_defect(
    aspect_ratio: float, solidity: float, extent: float, linearitas: float
) -> str:
    """Klasifikasi sederhana berdasarkan bentuk kontur.

    Heuristik (dijelaskan juga di README dan LAPORAN):
    - Goresan : memanjang dan lurus -> linearitas (PCA) sangat tinggi
                dan aspect ratio tinggi.
    - Retak   : memanjang tapi bergerigi / tidak beraturan -> linearitas
                lebih rendah, atau bentuknya tidak padat (solidity
                rendah) meski memanjang.
    - Noda    : bentuk membulat / tidak memanjang, aspect ratio ~1,
                linearitas rendah-menengah.
    """
    # Garis lurus yang padat (solidity tinggi) = goresan, walau aspect
    # ratio-nya sedikit di bawah 3 karena orientasi diagonal.
    # Retak juga bisa cukup "lurus" secara keseluruhan, tapi bentuknya
    # bergerigi sehingga solidity-nya rendah (< 0,70).
    if linearitas >= 0.95 and solidity >= 0.75 and aspect_ratio >= 2.5:
        return "goresan"
    if aspect_ratio >= 4.0 and solidity >= 0.75:
        return "goresan"
    if (solidity < 0.70 or extent < 0.35) and aspect_ratio >= 1.5:
        return "retak"
    if aspect_ratio >= 3.0 and linearitas < 0.90:
        return "retak"
    return "noda"


def linearity_pca(contour: np.ndarray) -> float:
    """Seberapa 'lurus' sebuah kontur, 0..1 (1 = garis lurus sempurna).

    Dihitung dari rasio eigenvalue terbesar PCA terhadap total variansi
    titik-titik kontur.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return 0.0
    pts = pts - pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    total = float(eig.sum())
    return float(eig[-1] / total) if total > 0 else 0.0


def segment_defects(
    image: np.ndarray,
    min_area: float = MIN_AREA,
    px_per_mm: float | None = None,
) -> tuple[np.ndarray, list[Defect], float]:
    """Segmentasi cacat dari citra BGR.

    Mengembalikan (mask_biner, daftar_cacat, nilai_otsu, ambang_yang_dipakai).
    Mask: cacat = putih (255), permukaan normal = hitam (0).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Gaussian blur 5x5 untuk menekan noise sensor.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Threshold Otsu + inversi: cacat (gelap) -> putih.
    #    Otsu memilih ambang otomatis dari histogram citra.
    #    PENGAMAN untuk citra bersih: histogram permukaan polos itu
    #    unimodal (hanya noise di sekitar nilai latar), sehingga Otsu
    #    akan membelahnya di tengah dan menganggap separuh permukaan
    #    sebagai "cacat". Karena itu ambang akhir dibatasi paling
    #    tinggi median_latar - 30: cacat asli jauh lebih gelap dari
    #    latar (selisih > 90), sedangkan noise hanya berselisih
    #    beberapa tingkat keabuan. Ambang yang dipakai dikembalikan
    #    agar bisa diaudit di JSON hasil.
    otsu_value, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    median_latar = float(np.median(blurred))
    threshold_used = min(float(otsu_value), median_latar - 30)
    _, mask = cv2.threshold(
        blurred, threshold_used, 255, cv2.THRESH_BINARY_INV
    )

    # 3. Morfologi: opening buang bintik noise, closing tutup lubang
    #    kecil di dalam cacat.
    # Kernel 3x3 (bukan 5x5): opening yang lebih besar akan menghapus
    # goresan tipis (lebar 2-3 px) yang justru ingin dideteksi.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Kontur + filter luas.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defects: list[Defect] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1, min(w, h))

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        extent = area / float(w * h) if w * h > 0 else 0.0

        lin = linearity_pca(contour)
        jenis = classify_defect(aspect_ratio, solidity, extent, lin)
        defects.append(
            Defect(
                id=len(defects) + 1,
                jenis=jenis,
                luas_px=round(area, 1),
                luas_mm2=(
                    round(area / (px_per_mm ** 2), 2)
                    if px_per_mm
                    else None
                ),
                bbox=[int(x), int(y), int(w), int(h)],
                aspect_ratio=round(aspect_ratio, 2),
                solidity=round(solidity, 2),
                extent=round(extent, 2),
                linearitas=round(lin, 2),
            )
        )

    # Urutkan dari cacat terbesar, supaya label #1 selalu yang dominan.
    defects.sort(key=lambda d: d.luas_px, reverse=True)
    for i, d in enumerate(defects, start=1):
        d.id = i

    return mask, defects, float(otsu_value), float(threshold_used)


def annotate(image: np.ndarray, defects: list[Defect]) -> np.ndarray:
    """Gambar bounding box + label pada salinan citra asli."""
    annotated = image.copy()
    for d in defects:
        x, y, w, h = d.bbox
        color = COLORS.get(d.jenis, (0, 255, 0))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"#{d.id} {d.jenis} ({d.luas_px:.0f}px)"
        cv2.putText(
            annotated,
            label,
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def detect_file(
    input_path: str | Path,
    min_area: float = MIN_AREA,
    px_per_mm: float | None = None,
) -> DetectionResult:
    """Deteksi cacat pada satu file citra, kembalikan ringkasannya."""
    path = Path(input_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Tidak bisa membaca citra: {path}")

    _, defects, otsu_value, threshold_used = segment_defects(image, min_area, px_per_mm)

    height, width = image.shape[:2]
    total_area = sum(d.luas_px for d in defects)
    percentage = round(total_area / (width * height) * 100, 2)

    return DetectionResult(
        file=path.name,
        ukuran_citra=[width, height],
        jumlah_cacat=len(defects),
        luas_total_cacat_px=round(total_area, 1),
        persentase_area_cacat=percentage,
        cacat=defects,
        threshold_otsu=round(otsu_value, 1),
        threshold_digunakan=round(threshold_used, 1),
    )


def process_file(
    input_path: str | Path,
    output_dir: str | Path,
    min_area: float = MIN_AREA,
    px_per_mm: float | None = None,
) -> tuple[DetectionResult, Path, Path]:
    """Deteksi satu citra dan simpan hasil anotasi + JSON ke output_dir.

    Mengembalikan (hasil, path_citra_anotasi, path_json).
    """
    path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Tidak bisa membaca citra: {path}")

    mask, defects, otsu_value, threshold_used = segment_defects(image, min_area, px_per_mm)
    height, width = image.shape[:2]
    total_area = sum(d.luas_px for d in defects)

    result = DetectionResult(
        file=path.name,
        ukuran_citra=[width, height],
        jumlah_cacat=len(defects),
        luas_total_cacat_px=round(total_area, 1),
        persentase_area_cacat=round(total_area / (width * height) * 100, 2),
        cacat=defects,
        threshold_otsu=round(otsu_value, 1),
        threshold_digunakan=round(threshold_used, 1),
    )

    annotated_path = output_dir / f"{path.stem}_annotated.png"
    json_path = output_dir / f"{path.stem}_summary.json"
    mask_path = output_dir / f"{path.stem}_mask.png"

    cv2.imwrite(str(annotated_path), annotate(image, defects))
    cv2.imwrite(str(mask_path), mask)
    json_path.write_text(result.to_json(), encoding="utf-8")

    return result, annotated_path, json_path
