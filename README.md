# AI Fact Checker - Analisis Teks Politik

Proyek ini adalah sebuah aplikasi web interaktif yang dibangun untuk menganalisis dan membandingkan performa berbagai model *machine learning* dalam tugas klasifikasi berita politik sebagai "Fakta" atau "Hoax". Aplikasi ini memungkinkan pengguna untuk memasukkan judul, kalimat, atau artikel pendek, lalu memilih salah satu dari empat model yang telah dilatih untuk melihat analisisnya secara *real-time*.

Tujuan utama proyek ini bukan hanya untuk membangun sebuah model, tetapi untuk melakukan analisis komparatif yang mendalam dan memahami kelebihan serta kekurangan dari setiap pendekatan, dari model statistik klasik hingga model Transformer modern.

## Fitur Utama
- **Analisis Teks Fleksibel:** Mampu menganalisis input berupa judul tunggal maupun artikel pendek dengan memecahnya per kalimat.
- **Perbandingan Model Interaktif:** Pengguna dapat memilih antara empat model yang berbeda (Naive Bayes, Logistic Regression, SVM, IndoBERT) untuk membandingkan hasilnya secara langsung.
- **Ringkasan Analisis:** Memberikan output yang jelas berupa probabilitas hoax rata-rata dan jumlah kalimat yang terdeteksi sebagai hoax.
- **Visualisasi per Kalimat:** Menyorot setiap kalimat dengan warna (hijau untuk Fakta, merah untuk Hoax) untuk memberikan pemahaman yang lebih mendalam tentang bagian mana dari teks yang dianggap mencurigakan oleh model.

## Model yang Dibandingkan
1.  **Multinomial Naive Bayes (MNB):** Model probabilistik klasik berbasis frekuensi kata.
2.  **Logistic Regression (LR):** Model linier untuk klasifikasi.
3.  **Support Vector Machine (SVM):** Model geometris yang mencari margin pemisah maksimal.
4.  **IndoBERT (Fine-Tuned):** Model Transformer modern yang memahami konteks bahasa Indonesia.

## Temuan Kunci & Kesimpulan
Eksperimen ini menghasilkan sebuah kesimpulan yang sangat penting dan tidak terduga:

> **Model terbaik secara statistik di atas kertas, belum tentu menjadi model terbaik dalam praktik.**

- **Juara Statistik:** **Multinomial Naive Bayes** secara konsisten mencapai F1-Score dan Akurasi tertinggi pada data uji yang berisi judul-judul berita individual.
- **Juara Praktis:** Namun, saat diuji pada artikel netral, Naive Bayes menjadi "paranoid" dan menghasilkan banyak *false positive*. Sebaliknya, **IndoBERT** menunjukkan pemahaman konteks yang superior, mampu membedakan penggunaan kata kunci politik dalam konteks netral, dan terbukti jauh lebih andal untuk penggunaan di dunia nyata.

Untuk aplikasi yang fungsional, **IndoBERT yang telah di-fine-tune** adalah model yang direkomendasikan.

## Teknologi yang Digunakan
- **Backend & ML:** Python
- **UI Framework:** Gradio
- **Deep Learning:** PyTorch, Transformers (Hugging Face)
- **Machine Learning Klasik:** Scikit-learn
- **Data Processing:** Pandas, NLTK, Sastrawi

## Setup dan Instalasi
Untuk menjalankan proyek ini di komputermu, ikuti langkah-langkah berikut dengan saksama.

**1. Clone Repositori (Jika ada)**
```bash
git clone [URL_REPO_ANDA]
cd [NAMA_FOLDER_PROYEK]
```

**2. Buat dan Aktifkan Virtual Environment (WAJIB!)**
Ini akan mengisolasi semua dependensi proyek dan mencegah konflik.
```bash
# Buat environment baru
python -m venv .venv

# Aktifkan environment (untuk Windows)
.venv\Scripts\activate
```

**3. Install PyTorch Versi GPU (Langkah Kritis)**
Instalasi PyTorch harus dilakukan secara manual untuk memastikan dukungan GPU (CUDA).
- Buka website resmi PyTorch: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- Pilih konfigurasi yang sesuai (Stable, Windows, Pip, Python, CUDA).
- Salin dan jalankan command yang dihasilkan. Contoh:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4. Install Dependensi Lainnya**
Setelah PyTorch terinstal, instal semua library lain menggunakan file `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Cara Menjalankan

Pastikan terminalmu berada di folder root proyek dan virtual environment sudah aktif `(.venv)`.

**Menjalankan Training (Opsional, jika ingin melatih ulang)**
```bash
# Contoh untuk melatih model Naive Bayes
python src/train_mb.py

# Contoh untuk melatih model IndoBERT
python src/train_indobert_comparison.py
```

**Menjalankan Aplikasi Utama (Gradio UI)**
```bash
python app.py
```
- Buka browser dan akses URL yang ditampilkan di terminal (biasanya `http://127.0.0.1:7860`).
- Aplikasi akan membutuhkan waktu beberapa saat untuk memuat semua model saat pertama kali dijalankan.

## Struktur Proyek
.
├── app.py                  # Skrip utama aplikasi Gradio
├── data/
│   └── raw/
│       └── dataset.csv     # Dataset final yang sudah bersih
├── models/
│   ├── classic_full/       # Model klasik hasil training data lengkap
│   └── indobert_full/      # Model IndoBERT hasil fine-tuning
├── src/
│   ├── train_mb.py         # Skrip training untuk setiap model
│   ├── train_lr.py
│   ├── train_svm.py
│   └── train_indobert_comparison.py
└── requirements.txt        # Daftar dependensi Python
