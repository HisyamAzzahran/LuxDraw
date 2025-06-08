# Gunakan base image Python 3.12 versi slim
FROM python:3.12-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Update daftar paket dan install dependency sistem yang dibutuhkan OpenCV
# Ini adalah bagian yang akan memperbaiki error libGL.so.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
# Hapus cache apt untuk menjaga ukuran image tetap kecil
&& rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt terlebih dahulu
COPY requirements.txt .

# Install semua dependency Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua sisa kode proyek Anda ke dalam container
COPY . .

# Perintahkan Railway untuk menjalankan aplikasi menggunakan Gunicorn
# Gunicorn akan berjalan di port yang disediakan oleh Railway
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]