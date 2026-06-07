# =============================================================
# PREPROCESSING DATA ULASAN TOKOPEDIA
# untuk: VADER, TextBlob, TF-IDF (SVM & RF), SparkMLlib, IndoBERT
# =============================================================

import pandas as pd
import re
import string
import unicodedata
import emoji
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── Load Data ─────────────────────────────────────────────────
df = pd.read_csv('data/tokopedia_baru.csv')
print(f"Data awal: {len(df)} baris")
print(f"Kolom: {df.columns.tolist()}")
print(f"\nDistribusi rating:\n{df['rating'].value_counts().sort_index()}")

# ════════════════════════════════════════════════════════════
# LANGKAH 1: LABELING SENTIMEN
# ════════════════════════════════════════════════════════════
def label_sentiment(rating):
    if rating <= 2:
        return 'negatif'
    elif rating == 3:
        return 'netral'
    else:
        return 'positif'

df['sentiment'] = df['rating'].apply(label_sentiment)
print(f"\nLabel sentimen:\n{df['sentiment'].value_counts()}")

# ════════════════════════════════════════════════════════════
# LANGKAH 2: HAPUS DATA KOTOR
# ════════════════════════════════════════════════════════════

# Hapus duplikat berdasarkan kolom ulasan
print(f"\nSebelum hapus duplikat: {len(df)}")
df = df.drop_duplicates(subset=['ulasan'])
print(f"Setelah hapus duplikat: {len(df)}")

# Hapus ulasan terlalu pendek (kurang dari 5 karakter)
df = df[df['ulasan'].astype(str).str.len() >= 5]
print(f"Setelah hapus ulasan pendek: {len(df)}")

# Reset index
df = df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════
# LANGKAH 3: KAMUS NORMALISASI SLANG
# ════════════════════════════════════════════════════════════
slang_dict = {
    # Negasi
    'gk':'tidak', 'ga':'tidak', 'gak':'tidak', 'ngga':'tidak',
    'nggak':'tidak', 'enggak':'tidak', 'ndak':'tidak', 'tdk':'tidak',
    # Kata umum
    'tp':'tapi', 'tpi':'tapi', 'krn':'karena', 'karna':'karena',
    'udah':'sudah', 'udh':'sudah', 'sdh':'sudah',
    'blm':'belum', 'blom':'belum', 'belom':'belum',
    'bgt':'banget', 'bngt':'banget',
    'lg':'lagi', 'lgi':'lagi',
    'msh':'masih', 'msih':'masih',
    'jg':'juga', 'jga':'juga',
    'jd':'jadi',
    'skrng':'sekarang', 'skrg':'sekarang', 'skr':'sekarang',
    'dg':'dengan', 'dgn':'dengan',
    'yg':'yang', 'dr':'dari', 'dlm':'dalam', 'utk':'untuk',
    'trus':'terus', 'truss':'terus',
    'klo':'kalau', 'klu':'kalau', 'kl':'kalau',
    'ad':'ada',
    # Produk & pengiriman
    'ori':'original', 'orgnl':'original',
    'cpt':'cepat', 'cpat':'cepat',
    'lmbt':'lambat',
    'packing':'kemasan', 'packaging':'kemasan', 'pckng':'kemasan',
    'seller':'penjual', 'buyer':'pembeli', 'shop':'toko',
    'fast':'cepat', 'slow':'lambat',
    'hp':'handphone',
    'sampe':'sampai', 'nyampe':'sampai',
    # Ekspresi positif
    'ok':'oke', 'okk':'oke', 'okee':'oke', 'okey':'oke',
    'mantep':'mantap', 'mantab':'mantap', 'mantull':'mantap',
    'mantul':'mantap', 'mantapp':'mantap', 'mantappp':'mantap',
    'keren':'bagus', 'bgs':'bagus', 'bgus':'bagus',
    'jos':'bagus', 'josss':'bagus',
    'good':'bagus', 'great':'bagus',
    'cakep':'bagus', 'apik':'bagus', 'sip':'bagus',
    'top':'bagus', 'recommended':'rekomen', 'rekomen':'rekomen',
    # Ekspresi negatif
    'jelek':'buruk', 'parah':'buruk',
    'zonk':'kecewa', 'brk':'rusak',
}

# ════════════════════════════════════════════════════════════
# LANGKAH 4: STOPWORDS BAHASA INDONESIA
# ════════════════════════════════════════════════════════════
factory_sw = StopWordRemoverFactory()
stopwords_id = set(factory_sw.get_stop_words())

# Tambah stopwords tambahan
custom_stopwords = {
    'aja', 'lho', 'sih', 'deh', 'dong', 'nih', 'tu', 'tuh',
    'kayak', 'kaya', 'kayaknya', 'emang', 'emg',
    'ih', 'ah', 'eh', 'yah', 'hah', 'wah', 'bah', 'lah', 'kah',
    'nya', 'ku', 'mu', 'lagi', 'juga', 'udah', 'udh'
}
stopwords_id.update(custom_stopwords)

# ════════════════════════════════════════════════════════════
# LANGKAH 5: FUNGSI PREPROCESSING
# ════════════════════════════════════════════════════════════
def hapus_emoji(text):
    """Hapus semua emoji dari teks."""
    return emoji.replace_emoji(text, replace='')

def normalisasi_slang(text):
    """Ganti kata tidak baku dengan kata baku."""
    words = text.split()
    return ' '.join([slang_dict.get(w, w) for w in words])

def clean_text(text):
    """
    Pembersihan teks utama:
    - lowercase
    - hapus emoji
    - hapus URL
    - hapus mention & hashtag
    - hapus angka
    - normalisasi unicode
    - hapus tanda baca
    - normalisasi huruf berulang (mantappp -> mantapp)
    - normalisasi spasi
    """
    text = str(text).lower()
    text = hapus_emoji(text)
    text = re.sub(r'http\S+|www\S+', '', text)       # hapus URL
    text = re.sub(r'@\w+|#\w+', '', text)             # hapus mention/hashtag
    text = re.sub(r'\d+', '', text)                    # hapus angka
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')  # hapus non-ASCII
    text = text.translate(str.maketrans('', '', string.punctuation))  # hapus tanda baca
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)        # normalisasi huruf berulang
    text = re.sub(r'\s+', ' ', text).strip()           # normalisasi spasi
    return text

def preprocess(text):
    """
    Pipeline preprocessing lengkap:
    clean -> normalisasi slang -> hapus stopwords
    """
    text = clean_text(text)
    text = normalisasi_slang(text)
    words = text.split()
    words = [w for w in words if w not in stopwords_id and len(w) > 1]
    return ' '.join(words)

def light_stem(text):
    """
    Stemming ringan rule-based (cepat, tanpa Sastrawi Stemmer).
    Hapus sufiks dan prefiks umum Bahasa Indonesia.
    """
    words = text.split()
    stemmed = []
    for word in words:
        w = re.sub(r'(kan|lah|kah|nya|ku|mu)$', '', word)
        if len(w) > 4:
            w = re.sub(r'an$', '', w)
        if len(w) > 4:
            w = re.sub(r'^(me|di|ke|se|ber|ter|pe)', '', w)
        if len(w) < 2:
            w = word
        stemmed.append(w)
    return ' '.join(stemmed)

# ════════════════════════════════════════════════════════════
# LANGKAH 6: JALANKAN PREPROCESSING
# ════════════════════════════════════════════════════════════
print("\nMemproses clean_text...")
df['clean_text'] = df['ulasan'].apply(preprocess)

print("Memproses stemmed_text...")
df['stemmed_text'] = df['clean_text'].apply(light_stem)

# Hapus hasil preprocessing yang kosong
df = df[df['clean_text'].str.strip().str.len() > 0]
df = df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════
# LANGKAH 7: LABEL NUMERIK
# ════════════════════════════════════════════════════════════
label_map = {'negatif': 0, 'netral': 1, 'positif': 2}
df['label'] = df['sentiment'].map(label_map)

# ════════════════════════════════════════════════════════════
# LANGKAH 8: SIMPAN HASIL
# ════════════════════════════════════════════════════════════
output_cols = ['ulasan', 'clean_text', 'stemmed_text', 'rating', 'sentiment', 'label']
df[output_cols].to_csv('data_preprocessed_final.csv', index=False)

# ════════════════════════════════════════════════════════════
# RINGKASAN HASIL
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("HASIL PREPROCESSING")
print("="*55)
print(f"Total data final  : {len(df):,} baris")
print(f"\nDistribusi label:")
print(df['sentiment'].value_counts().to_string())
print(f"\nNull values:")
print(df[output_cols].isnull().sum().to_string())
print(f"\nContoh hasil preprocessing:")
for _, row in df.sample(3, random_state=42).iterrows():
    print(f"\n  [Rating {row['rating']} → {row['sentiment']}]")
    print(f"  Asli    : {str(row['ulasan'])[:70]}")
    print(f"  Clean   : {str(row['clean_text'])[:70]}")
    print(f"  Stemmed : {str(row['stemmed_text'])[:70]}")

print("\n File tersimpan: data_preprocessed_final.csv")
print("\nKolom tersedia:")
print("  • ulasan        → teks asli (untuk VADER & TextBlob)")
print("  • clean_text    → cleaned (untuk TF-IDF SVM/RF & IndoBERT)")
print("  • stemmed_text  → stemmed (untuk SparkMLlib & TF-IDF RF)")
print("  • sentiment     → label string: negatif/netral/positif")
print("  • label         → label numerik: 0/1/2")