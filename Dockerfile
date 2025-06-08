FROM python:3.12-slim as builder

# Install easyocr saja, karena hanya itu yang kita butuhkan untuk download
RUN pip install easyocr

# Jalankan perintah Python yang akan men-trigger download model.
# Model akan disimpan di direktori default /root/.EasyOCR/
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"


# --- Tahap 2: Final Image ---
# Ini adalah image akhir yang akan dijalankan oleh Railway
FROM python:3.12-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Update daftar paket dan install dependency sistem yang dibutuhkan OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
# Hapus cache apt untuk menjaga ukuran image tetap kecil
&& rm -rf /var/lib/apt/lists/*

# ---- PERBAIKAN UTAMA ADA DI SINI ----
# Salin model yang sudah diunduh dari tahap 'builder' ke dalam image ini
COPY --from=builder /root/.EasyOCR /root/.EasyOCR

# Salin file requirements.txt
COPY requirements.txt .

# Install semua dependency Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua sisa kode proyek Anda
COPY . .

# Perintah untuk menjalankan aplikasi. Ini tidak berubah.
CMD gunicorn --bind 0.0.0.0:$PORT "app:create_app()"