# Laporan Tugas Presentasi Individu

## Deteksi Cacat Permukaan Produk Menggunakan Segmentasi Citra

**Mata kuliah:** Pengolahan Citra Digital (22TIF0262) — Teknik Informatika S1, Semester V
**Tema:** D — Industri dan Manufaktur
**Nama:** Tamir
**Output:** Video presentasi (≥ 10 menit) + source code Python + laporan ini

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Quality control (QC) adalah tahap penting dalam industri manufaktur:
produk yang cacat — tergores, bernoda, atau retak — tidak boleh lolos
ke tangan konsumen. Pemeriksaan manual oleh manusia bersifat lambat,
melelahkan, dan tidak konsisten; cacat kecil mudah terlewat, terutama
pada jalur produksi yang berjalan cepat.

Pengolahan citra digital menawarkan pemeriksaan yang otomatis, cepat,
dan objektif. Dengan kamera dan program komputer, permukaan produk
dapat diperiksa satu per satu dengan kriteria yang sama setiap waktu.
Tugas ini membangun sistem tersebut dalam skala kecil: sebuah program
Python yang menerima citra permukaan produk, menemukan area cacat
melalui segmentasi, lalu mengukur dan mengklasifikasikannya.

### 1.2 Rumusan Masalah

1. Bagaimana memisahkan area cacat dari permukaan produk yang normal
   pada sebuah citra?
2. Bagaimana mengukur tingkat keparahan cacat (jumlah, luas, dan
   persentase area)?
3. Bagaimana membedakan jenis cacat — goresan, noda, dan retak —
   berdasarkan bentuknya?

### 1.3 Tujuan

Membangun program Python yang dapat mendeteksi, mengukur, dan
mengklasifikasikan cacat permukaan produk dari citra, serta
menampilkan hasilnya dalam bentuk citra teranotasi dan ringkasan
kuantitatif (JSON/CSV).

---

## 2. Landasan Teori

**Grayscale dan Gaussian blur.** Citra warna disederhanakan ke satu
kanal intensitas (grayscale), lalu dihaluskan dengan filter Gaussian
5×5 untuk menekan noise sensor yang dapat disalahartikan sebagai cacat.

**Threshold Otsu.** Metode thresholding otomatis yang memilih ambang
dari histogram citra dengan memaksimalkan variansi antar-kelas
(latar vs. objek). Cocok di sini karena cacat tampak jauh lebih gelap
daripada permukaan produk yang terang. Ambang akhir dibatasi pada
`min(Otsu, median_latar − 30)` agar citra bersih (histogram unimodal)
tidak terbelah Otsu menjadi "cacat" palsu.

**Operasi morfologi.** Opening (erosi lalu dilasi) membuang bintik
noise kecil; closing (dilasi lalu erosi) menutup lubang kecil di dalam
area cacat. Kernel yang dipakai berukuran 3×3 agar goresan tipis
(lebar 2–3 px) tidak ikut terhapus.

**Deteksi kontur dan deskriptor bentuk.** Setiap area putih pada mask
biner diambil konturnya (`findContours`). Dari tiap kontur dihitung:
luas, bounding box, *aspect ratio* (perbandingan sisi terpanjang dan
terpendek), *solidity* (luas kontur / luas convex hull), *extent*
(luas kontur / luas bounding box), dan *linearitas* (rasio eigenvalue
terbesar PCA terhadap total variansi — 1 berarti garis lurus sempurna).
Kombinasi deskriptor ini dipakai untuk mengklasifikasikan cacat:
goresan (memanjang dan lurus), retak (memanjang tapi bergerigi,
solidity rendah), dan noda (membulat, aspect ratio ≈ 1).

---

## 3. Metodologi

### 3.1 Pipeline

```
Citra input → Grayscale → Gaussian blur → Threshold Otsu (+ pengaman)
→ Opening → Closing → Deteksi kontur → Filter luas (≥ 40 px)
→ Pengukuran & klasifikasi → Anotasi (bounding box + label)
→ Output: citra teranotasi, mask, JSON, CSV
```

Implementasi: `src/detect.py` (pipeline dan klasifikasi) dan
`src/main.py` (CLI). Pustaka yang digunakan: OpenCV, NumPy, Matplotlib.

### 3.2 Data

Karena tidak ada dataset cacat nyata yang tersedia, dibuat generator
citra sintetis (`scripts/generate_samples.py`): pelat logam/keramik
polos berwarna terang dengan noise halus, lalu digambar cacat
(goresan, noda, retak) secara deterministik. Total 5 citra 800×600 px:
2 bersih (`bersih_01`, `bersih_02`) dan 3 bercacat (`cacat_01`–`cacat_03`)
dengan jumlah dan jenis cacat yang diketahui, sehingga hasil deteksi
dapat dinilai kebenarannya.

### 3.3 Cara Menjalankan

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_samples.py
python -m src.main --input data/samples/cacat_01.png --output results/
python -m src.main --input data/samples/ --output results/
```

---

## 4. Hasil dan Analisis

Hasil deteksi pada seluruh sampel (perintah di atas):

| Citra | Cacat sebenarnya | Terdeteksi | Rincian | % area cacat |
|---|---|---|---|---|
| bersih_01.png | 0 | 0 | – | 0,00% |
| bersih_02.png | 0 | 0 | – | 0,00% |
| cacat_01.png | 3 | 3 | goresan ×2, noda ×1 | 1,93% |
| cacat_02.png | 4 | 4 | retak ×1, noda ×2, goresan ×1 | 2,28% |
| cacat_03.png | 3 | 3 | retak ×1, noda ×1, goresan ×1 | 3,22% |

**10 dari 10 cacat terdeteksi dan jenisnya semuanya benar.**
Kedua citra bersih menghasilkan 0 cacat — tidak ada false positive —
yang menunjukkan pengaman ambang (`median_latar − 30`) bekerja.
Persentase area cacat naik seiring jumlah dan ukuran cacat
(1,93% → 2,28% → 3,22%), sehingga metrik ini dapat dipakai sebagai
indikator keparahan dan ambang lolos/tidak-lolos QC (mis. produk
ditolak jika area cacat > 3%).

Dua masalah yang ditemukan saat pengujian dan cara mengatasinya:
(1) Otsu murni pada citra bersih menghasilkan ratusan "cacat" palsu —
diatasi dengan membatasi ambang pada `median_latar − 30`;
(2) opening 5×5 menghapus goresan tipis 2 px — diatasi dengan kernel
3×3. Kedua perbaikan ini didokumentasikan di `src/detect.py`.

Contoh hasil visual: `results/cacat_01_annotated.png` — dua kotak
kuning (goresan) dan satu kotak merah (noda) dengan label luasnya.

---

## 5. Kesimpulan

1. Pipeline segmentasi (grayscale → Gaussian blur → Otsu → morfologi
   → kontur) berhasil mendeteksi seluruh cacat pada citra sampel
   sintetis, tanpa false positive pada citra bersih.
2. Deskriptor bentuk (aspect ratio, solidity, extent, linearitas PCA)
   cukup untuk membedakan goresan, retak, dan noda pada data uji ini.
3. Sistem menghasilkan output yang siap dipakai untuk QC sederhana:
   citra teranotasi, mask segmentasi, serta ringkasan JSON/CSV berisi
   jumlah cacat, luas total, dan persentase area cacat.

**Saran pengembangan:** uji pada foto produk nyata dengan pencahayaan
tidak merata (adaptive threshold / CLAHE), kalibrasi skala piksel-ke-mm
agar luas dilaporkan dalam mm², dan klasifikasi berbasis machine
learning setelah dataset cacat nyata terkumpul.

---

## Referensi

1. Panduan Tugas Presentasi, Pengolahan Citra Digital (22TIF0262),
   Teknik Informatika S1 Semester V.
2. OpenCV Documentation — Image Thresholding, Morphological
   Transformations, Contours. https://docs.opencv.org/
3. Gonzalez, R. C. & Woods, R. E., *Digital Image Processing*.
4. scikit-image Documentation. https://scikit-image.org/
