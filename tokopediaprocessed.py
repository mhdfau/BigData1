import pandas as pd
import re
import os

# ══════════════════════════════════════════════
# BACA DATA MENTAH
# ══════════════════════════════════════════════
df_mentah = pd.read_csv("tokopedia_baru.csv")                            # Baca file CSV asli hasil scraping
print(f"Data mentah dimuat: {df_mentah.shape[0]} baris, {df_mentah.shape[1]} kolom")

# ══════════════════════════════════════════════
# BUAT SALINAN UNTUK PROCESSING
# ══════════════════════════════════════════════
df_processed = df_mentah.copy()                                           # Salin data mentah, asli tidak diubah

# ══════════════════════════════════════════════
# FUNGSI PEMBERSIHAN TEKS LENGKAP
# ══════════════════════════════════════════════
def bersihkan_teks(teks):
    teks = str(teks)                                                      # Pastikan bertipe string
    teks = re.sub(r'[^\x00-\x7F]', '', teks)                            # Hapus emoji & karakter non-ASCII
    teks = teks.lower()                                                   # Ubah semua huruf jadi kecil
    teks = re.sub(r'http\S+|www\S+', '', teks)                          # Hapus URL (http://... atau www....)
    teks = re.sub(r'@\w+', '', teks)                                     # Hapus mention (@username)
    teks = re.sub(r'#\w+', '', teks)                                     # Hapus hashtag (#kata)
    teks = re.sub(r'\d+', '', teks)                                      # Hapus semua angka
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)                             # Hapus tanda baca & simbol tersisa
    teks = re.sub(r'\s+', ' ', teks).strip()                            # Hapus spasi ganda & spasi di ujung
    return teks

# ══════════════════════════════════════════════
# TERAPKAN PEMBERSIHAN KE KOLOM ULASAN
# ══════════════════════════════════════════════
print(" Membersihkan teks ulasan...")
df_processed['ulasan_bersih'] = df_processed['ulasan'].apply(bersihkan_teks)  # Terapkan fungsi ke semua ulasan

# ══════════════════════════════════════════════
# TAMBAH KOLOM SENTIMEN DARI RATING
# ══════════════════════════════════════════════
df_processed['sentimen'] = df_processed['rating'].apply(                 # Buat kolom sentimen berdasarkan rating
    lambda x: 'positif' if x >= 4 else ('netral' if x == 3 else 'negatif')  # >= 4 positif, 3 netral, <= 2 negatif
)

# ══════════════════════════════════════════════
# TAMBAH KOLOM PANJANG ULASAN
# ══════════════════════════════════════════════
df_processed['panjang_ulasan'] = df_processed['ulasan_bersih'].apply(   # Hitung jumlah kata dari ulasan BERSIH
    lambda x: len(str(x).split())                                         # Split per spasi lalu hitung
)

# ══════════════════════════════════════════════
# CEK HASIL PEMBERSIHAN (SAMPLE)
# ══════════════════════════════════════════════
print("\n Contoh hasil pembersihan:")
print("-" * 60)
for i in range(3):                                                        # Tampilkan 3 contoh
    print(f"SEBELUM : {df_processed['ulasan'].iloc[i][:100]}")           # Teks asli (100 karakter pertama)
    print(f"SESUDAH : {df_processed['ulasan_bersih'].iloc[i][:100]}")   # Teks bersih (100 karakter pertama)
    print("-" * 60)

# ══════════════════════════════════════════════
# INFO DISTRIBUSI SENTIMEN
# ══════════════════════════════════════════════
print("\n Distribusi sentimen:")
print(df_processed['sentimen'].value_counts().to_string())               # Tampilkan jumlah per sentimen

print("\n Distribusi per kategori & sentimen:")
print(df_processed.groupby(['kategori', 'sentimen']).size()             # Hitung per kategori dan sentimen
      .reset_index(name='jumlah').to_string(index=False))

# ══════════════════════════════════════════════
# SIMPAN DUA FILE TERPISAH
# ══════════════════════════════════════════════
df_mentah.to_csv("tokopedia_mentah.csv", index=False)                   # Simpan data mentah (tidak diubah)
df_processed.to_csv("tokopedia_processed.csv", index=False)             # Simpan data hasil processing

print("\n Selesai!")
print(f"   Data mentah    : {df_mentah.shape[0]} baris, {df_mentah.shape[1]} kolom → tokopedia_mentah.csv")
print(f"   Data processed : {df_processed.shape[0]} baris, {df_processed.shape[1]} kolom → tokopedia_processed.csv")
print(f"\n   Kolom baru yang ditambahkan:")
print(f"   - ulasan_bersih  : teks ulasan sudah dibersihkan")
print(f"   - sentimen       : positif / netral / negatif")
print(f"   - panjang_ulasan : jumlah kata per ulasan")