# Deteksi Cacat Permukaan Produk Menggunakan Segmentasi Citra

Tugas Presentasi Individu — **Pengolahan Citra Digital (22TIF0262)**,
Program Studi Teknik Informatika S1, Semester V.
Tema D: **Industri dan Manufaktur**.

> Sistem ini mendeteksi cacat pada permukaan produk (goresan, noda, dan
> retak) dari sebuah citra, lalu menandai tiap cacat dengan bounding box,
> menghitung jumlah cacat, luas total cacat, dan persentase area cacat.

## Latar Belakang / Motivasi

Dalam industri manufaktur, pemeriksaan kualitas produk (quality control)
masih sering dilakukan secara manual oleh manusia. Cara ini lambat,
melelahkan, dan hasilnya tidak konsisten — cacat kecil seperti goresan
tipis atau noda samar mudah terlewat, apalagi pada jalur produksi yang
cepat.

Pengolahan citra digital menawarkan solusi: kamera memotret permukaan
produk, lalu program menganalisis citra tersebut secara otomatis dan
objektif. Tugas ini membangun pipeline sederhana namun lengkap untuk
itu: segmentasi memisahkan area cacat (yang tampak lebih gelap dari
permukaan normal), kemudian setiap area diukur dan diklasifikasikan
berdasarkan bentuknya.

## Metode / Pipeline

Alur pemrosesan di `src/detect.py`, langkah demi langkah:

| Langkah | Operasi | Tujuan |
|---|---|---|
| 1 | Baca citra → **grayscale** | Sederhanakan citra ke satu kanal intensitas |
| 2 | **Gaussian blur 5×5** | Menekan noise sensor agar tidak terdeteksi sebagai cacat |
| 3 | **Threshold Otsu** (dibalik) | Memisahkan cacat yang gelap dari permukaan terang secara otomatis, tanpa ambang yang ditentukan manual |
| 4 | **Pengaman ambang**: ambang akhir = `min(Otsu, median_latar − 30)` | Mencegah citra bersih terdeteksi sebagai cacat — histogram permukaan polos itu unimodal, Otsu murni akan membelahnya di tengah |
| 5 | **Morfologi opening** (kernel 3×3) | Membuang bintik noise kecil |
| 6 | **Morfologi closing** (kernel 3×3, 2 iterasi) | Menutup lubang kecil di dalam cacat |
| 7 | **Deteksi kontur** (`findContours`) | Mendapatkan tiap area cacat sebagai satu objek |
| 8 | **Filter luas** (min. 40 px) | Membuang sisa noise yang sangat kecil |
| 9 | **Pengukuran & klasifikasi** | Luas, bounding box, aspect ratio, solidity, extent, linearitas (PCA); klasifikasi goresan / retak / noda |
| 10 | **Anotasi** | Bounding box + label pada citra asli |

### Klasifikasi cacat (heuristik bentuk)

- **Goresan** — memanjang dan lurus: linearitas PCA ≥ 0,95, solidity
  ≥ 0,75, dan aspect ratio ≥ 2,5. Contoh: garis lurus tipis.
- **Retak** — memanjang tapi bergerigi / tidak beraturan: solidity
  rendah (< 0,70) atau extent rendah (< 0,35) pada bentuk yang
  memanjang. Contoh: garis patah-patah.
- **Noda** — membulat, tidak memanjang: aspect ratio mendekati 1,
  solidity mendekati 1. Contoh: bercak gelap berbentuk elips.

Warna anotasi: goresan = kuning, retak = magenta, noda = merah.

## Struktur Repo

```
deteksi-cacat-permukaan-produk/
├── src/
│   ├── __init__.py      # ekspor detect_file / process_file
│   ├── detect.py        # pipeline segmentasi + klasifikasi + anotasi
│   └── main.py          # CLI demo (satu citra atau satu folder)
├── scripts/
│   └── generate_samples.py   # generator citra sampel sintetis
├── data/
│   └── samples/         # 5 citra sampel (dibuat generator, 800×600)
│       ├── bersih_01.png, bersih_02.png   # tanpa cacat
│       ├── cacat_01.png  # 2 goresan + 1 noda
│       ├── cacat_02.png  # 1 retak + 2 noda + 1 goresan
│       └── cacat_03.png  # 1 retak + 1 goresan + 1 noda besar
├── results/             # hasil deteksi (dibuat saat menjalankan demo)
│   ├── *_annotated.png  # citra asli + bounding box & label
│   ├── *_mask.png       # mask biner hasil segmentasi
│   ├── *_summary.json   # ringkasan per citra
│   └── summary.csv      # ringkasan semua citra
├── requirements.txt
├── .gitignore
├── README.md            # dokumen ini
└── LAPORAN.md           # kerangka laporan tugas (2–3 halaman)
```

## Instalasi

Python 3.10+ (diuji dengan Python 3.12).

```bash
# 1. Buat virtual environment (disarankan)
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 2. Instal dependensi
pip install -r requirements.txt
```

Dependensi: `opencv-python`, `numpy`, `matplotlib` (matplotlib saat ini
dicantumkan untuk kebutuhan visualisasi/analisis lanjutan).

## Cara Menjalankan Demo

### 1. Buat citra sampel (opsional — sampel sudah tersedia di `data/samples/`)

```bash
python scripts/generate_samples.py
```

Generator membuat permukaan pelat logam/keramik polos dengan noise halus,
lalu menggambar cacat sintetis (goresan, noda, retak) secara deterministik
(seed tetap), sehingga demo berjalan sepenuhnya offline tanpa dataset
eksternal.

### 2. Deteksi satu citra

```bash
python -m src.main --input data/samples/cacat_01.png --output results/
```

### 3. Deteksi semua sampel sekaligus

```bash
python -m src.main --input data/samples/ --output results/
```

Opsi tambahan:

```bash
# Abaikan kontur yang luasnya di bawah 80 px
python -m src.main --input data/samples/ --output results/ --min-area 80

# Konversi luas cacat ke mm² (jika diketahui skala, mis. 5 px = 1 mm)
python -m src.main --input data/samples/cacat_01.png --output results/ --px-per-mm 5
```

## Contoh Hasil

Hasil uji pada 5 citra sampel (800×600 px), perintah
`python -m src.main --input data/samples/ --output results/`:

| Citra | Cacat sebenarnya | Terdeteksi | Rincian | Luas total | % area |
|---|---|---|---|---|---|
| `bersih_01.png` | 0 | **0** | – | 0 px | 0,00% |
| `bersih_02.png` | 0 | **0** | – | 0 px | 0,00% |
| `cacat_01.png` | 3 | **3** | goresan ×2, noda ×1 | 9.258 px | 1,93% |
| `cacat_02.png` | 4 | **4** | retak ×1, noda ×2, goresan ×1 | 10.925 px | 2,28% |
| `cacat_03.png` | 3 | **3** | retak ×1, noda ×1, goresan ×1 | 15.458 px | 3,22% |

**10 dari 10 cacat terdeteksi, dengan jenis yang semuanya benar.**
Citra bersih menghasilkan 0 cacat — tidak ada false positive.

Contoh output terminal:

```
cacat_01.png: 3 cacat (noda x1, goresan x2) | luas total 9258 px (1.93% area) | Otsu=150.0
  -> results/cacat_01_annotated.png
  -> results/cacat_01_summary.json
```

Setiap citra menghasilkan tiga file di `results/`:

- `*_annotated.png` — citra asli dengan bounding box dan label
  (`#1 noda (6618px)`, `#2 goresan (1484px)`, …)
- `*_mask.png` — mask biner hasil segmentasi (cacat = putih)
- `*_summary.json` — ringkasan lengkap: jumlah cacat, luas total,
  persentase area, nilai threshold Otsu dan ambang yang dipakai,
  serta detail tiap cacat (jenis, luas, bounding box, aspect ratio,
  solidity, extent, linearitas)

Ringkasan semua citra juga tersedia di `results/summary.csv`.

## Keterbatasan & Pengembangan Lanjutan

- Klasifikasi masih berupa heuristik bentuk; pada citra nyata dengan
  pencahayaan tidak merata, adaptive threshold atau CLAHE dapat
  membantu sebelum threshold Otsu.
- Ambang `median_latar − 30` dan luas minimum 40 px disetel untuk
  permukaan terang yang seragam; untuk material gelap atau bertekstur,
  parameter ini perlu dikalibrasi ulang.
- Pengembangan lanjutan: klasifikasi dengan machine learning
  (mis. SVM / CNN) setelah dataset cacat nyata terkumpul, serta
  kalibrasi skala piksel-ke-mm agar luas cacat dapat dilaporkan
  dalam satuan fisik.

## Referensi

1. Panduan Tugas Presentasi, Pengolahan Citra Digital (22TIF0262),
   Teknik Informatika S1 Semester V — Tema D: Industri dan Manufaktur.
2. OpenCV Documentation — Image Thresholding, Morphological
   Transformations, Contours: https://docs.opencv.org/
3. Gonzalez, R. C. & Woods, R. E., *Digital Image Processing* —
   thresholding (Otsu), morfologi, dan analisis kontur.
4. scikit-image Documentation: https://scikit-image.org/
