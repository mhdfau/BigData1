# ============================================================
# METODE 1 — VADER (NLTK)
# METODE 2 — TextBlob
# ============================================================
# Mata Kuliah : Big Data — Topik 3
# Deskripsi   : Analisis sentimen berbasis leksikon
#               VADER menggunakan kamus sentimen English
#               TextBlob menghasilkan Polarity & Subjectivity
# Cara jalankan: python 1_vader_textblob.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
import nltk
import time, json, os, warnings
warnings.filterwarnings('ignore')

# Download resource NLTK
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

os.makedirs('hasil', exist_ok=True)

print("=" * 60)
print("   METODE 1 & 2 — VADER + TextBlob")
print("   Big Data Topik 3 ")
print("=" * 60)

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
df = pd.read_csv("data/tokopedia_processed.csv")
df = df.dropna(subset=['ulasan_bersih'])
df = df[df['panjang_ulasan'] >= 3].copy().reset_index(drop=True)

print(f"\n Dataset dimuat: {len(df):,} ulasan")
print(f"\n   Distribusi Sentimen:")
for s, n in df['sentimen'].value_counts().items():
    bar = " " * int(n/100)
    print(f"   {s:10s}: {n:5,}  {bar}")

# ──────────────────────────────────────────────────────────
# FUNGSI EVALUASI LENGKAP
# ──────────────────────────────────────────────────────────
LABELS = ['positif', 'negatif', 'netral']

def evaluasi_lengkap(y_true, y_pred, nama, waktu):
    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    pre = round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
    rec = round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
    f1  = round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
    cm  = confusion_matrix(y_true, y_pred, labels=LABELS)

    print(f"\n{'─'*55}")
    print(f"   HASIL EVALUASI — {nama}")
    print(f"{'─'*55}")
    print(f"  Accuracy         : {acc}%")
    print(f"  Precision        : {pre}%")
    print(f"  Recall           : {rec}%")
    print(f"  F1-Score         : {f1}%")
    print(f"  Processing Time  : {waktu} detik")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
          target_names=LABELS, zero_division=0))

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(pd.DataFrame(cm, index=LABELS, columns=LABELS),
                annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, linecolor='gray')
    plt.title(f'Confusion Matrix — {nama}', fontsize=13, pad=12)
    plt.ylabel('Label Sebenarnya', fontsize=11)
    plt.xlabel('Label Prediksi', fontsize=11)
    plt.tight_layout()
    fname = nama.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'hasil/cm_{fname}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrix disimpan: hasil/cm_{fname}.png")

    return dict(accuracy=acc, precision=pre, recall=rec,
                f1=f1, waktu=waktu, cm=cm.tolist())

# ──────────────────────────────────────────────────────────
# METODE 1 — VADER
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  METODE 1 — VADER (NLTK)")
print("=" * 60)
print("""
  Cara kerja VADER:
  Teks → SentimentIntensityAnalyzer
       → Skor: neg, neu, pos, compound
  
  Klasifikasi compound:
  compound >=  0.05 → Positif
  compound <= -0.05 → Negatif
  sisanya           → Netral
""")

sia = SentimentIntensityAnalyzer()

def vader_predict(teks):
    skor     = sia.polarity_scores(str(teks))
    compound = skor['compound']
    if   compound >=  0.05: label = 'positif'
    elif compound <= -0.05: label = 'negatif'
    else:                   label = 'netral'
    return label, skor['compound'], skor['neg'], skor['neu'], skor['pos']

print(" Menjalankan VADER pada seluruh dataset...")
t0 = time.time()
hasil_v = df['ulasan_bersih'].apply(vader_predict)
t_vader = round(time.time() - t0, 2)

df['vader_label']    = [h[0] for h in hasil_v]
df['vader_compound'] = [h[1] for h in hasil_v]
df['vader_neg']      = [h[2] for h in hasil_v]
df['vader_neu']      = [h[3] for h in hasil_v]
df['vader_pos']      = [h[4] for h in hasil_v]

print(f" Selesai: {t_vader} detik\n")
print("  Contoh hasil VADER (5 sampel acak):")
print(f"  {'Ulasan (55 karakter)':<57}{'Compound':>10}{'Prediksi':>10}{'Aktual':>10}")
print(f"  {'─'*57}{'─'*10}{'─'*10}{'─'*10}")
for _, r in df.sample(5, random_state=7).iterrows():
    print(f"  {str(r['ulasan_bersih'])[:55]:<57}"
          f"{r['vader_compound']:>10.3f}"
          f"{r['vader_label']:>10}"
          f"{r['sentimen']:>10}")

eval_vader = evaluasi_lengkap(df['sentimen'], df['vader_label'], 'VADER NLTK', t_vader)

# ──────────────────────────────────────────────────────────
# METODE 2 — TextBlob
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  METODE 2 — TextBlob")
print("=" * 60)
print("""
  Cara kerja TextBlob:
  Teks → TextBlob(teks).sentiment
       → Polarity    : -1.0 s/d +1.0
       → Subjectivity:  0.0 s/d  1.0

  Klasifikasi polarity:
  polarity >  0.05 → Positif
  polarity < -0.05 → Negatif
  sisanya          → Netral
""")

def textblob_predict(teks):
    blob = TextBlob(str(teks))
    pol  = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    if   pol >  0.05: label = 'positif'
    elif pol < -0.05: label = 'negatif'
    else:             label = 'netral'
    return label, pol, subj

print(" Menjalankan TextBlob pada seluruh dataset...")
t0 = time.time()
hasil_tb = df['ulasan_bersih'].apply(textblob_predict)
t_tb = round(time.time() - t0, 2)

df['tb_label']        = [h[0] for h in hasil_tb]
df['tb_polarity']     = [h[1] for h in hasil_tb]
df['tb_subjectivity'] = [h[2] for h in hasil_tb]

print(f" Selesai: {t_tb} detik\n")
print("  Contoh hasil TextBlob (5 sampel acak):")
print(f"  {'Ulasan (50 karakter)':<52}{'Polarity':>10}{'Subj':>8}{'Prediksi':>10}{'Aktual':>10}")
print(f"  {'─'*52}{'─'*10}{'─'*8}{'─'*10}{'─'*10}")
for _, r in df.sample(5, random_state=7).iterrows():
    print(f"  {str(r['ulasan_bersih'])[:50]:<52}"
          f"{r['tb_polarity']:>10.3f}"
          f"{r['tb_subjectivity']:>8.3f}"
          f"{r['tb_label']:>10}"
          f"{r['sentimen']:>10}")

eval_tb = evaluasi_lengkap(df['sentimen'], df['tb_label'], 'TextBlob', t_tb)

# ──────────────────────────────────────────────────────────
# PERBANDINGAN VADER vs TextBlob
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PERBANDINGAN VADER vs TextBlob")
print("=" * 60)

tabel_perb = pd.DataFrame({
    'Metode'           : ['VADER (NLTK)', 'TextBlob'],
    'Accuracy (%)'     : [eval_vader['accuracy'],  eval_tb['accuracy']],
    'Precision (%)'    : [eval_vader['precision'], eval_tb['precision']],
    'Recall (%)'       : [eval_vader['recall'],    eval_tb['recall']],
    'F1-Score (%)'     : [eval_vader['f1'],        eval_tb['f1']],
    'Processing (detik)': [eval_vader['waktu'],    eval_tb['waktu']],
})
print(tabel_perb.to_string(index=False))
print("\n    Catatan: Akurasi rendah wajar karena VADER & TextBlob")
print("     dirancang untuk Bahasa Inggris, bukan Bahasa Indonesia.")

# Visualisasi perbandingan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
metrik = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
v_vals = [eval_vader['accuracy'], eval_vader['precision'],
          eval_vader['recall'],   eval_vader['f1']]
t_vals = [eval_tb['accuracy'],    eval_tb['precision'],
          eval_tb['recall'],      eval_tb['f1']]
x = np.arange(len(metrik)); w = 0.35
b1 = ax1.bar(x-w/2, v_vals, w, label='VADER', color='#27AE60', alpha=0.85)
b2 = ax1.bar(x+w/2, t_vals, w, label='TextBlob', color='#2980B9', alpha=0.85)
ax1.set_title('Perbandingan VADER vs TextBlob', fontsize=12)
ax1.set_xticks(x); ax1.set_xticklabels(metrik)
ax1.set_ylim(0, 80); ax1.set_ylabel('Nilai (%)')
ax1.legend(); ax1.grid(axis='y', alpha=0.3)
ax1.bar_label(b1, fmt='%.1f%%', padding=3, fontsize=8)
ax1.bar_label(b2, fmt='%.1f%%', padding=3, fontsize=8)

# Distribusi prediksi
cats = ['positif', 'negatif', 'netral']
ax2.bar(np.arange(3)-0.2, [df['vader_label'].eq(c).sum() for c in cats],
        0.4, label='VADER', color='#27AE60', alpha=0.85)
ax2.bar(np.arange(3)+0.2, [df['tb_label'].eq(c).sum() for c in cats],
        0.4, label='TextBlob', color='#2980B9', alpha=0.85)
ax2.bar(np.arange(3)+0.6, [df['sentimen'].eq(c).sum() for c in cats],
        0.4, label='Aktual', color='#E74C3C', alpha=0.85)
ax2.set_title('Distribusi Prediksi vs Aktual', fontsize=12)
ax2.set_xticks(np.arange(3)); ax2.set_xticklabels(cats)
ax2.set_ylabel('Jumlah Data'); ax2.legend(); ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('hasil/perbandingan_vader_textblob.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   Grafik disimpan: hasil/perbandingan_vader_textblob.png")

# ──────────────────────────────────────────────────────────
# SIMPAN
# ──────────────────────────────────────────────────────────
df.to_csv('hasil/hasil_m1_m2.csv', index=False, encoding='utf-8-sig')
with open('hasil/eval_m1_m2.json', 'w') as f:
    json.dump({'VADER': eval_vader, 'TextBlob': eval_tb}, f, indent=2, default=str)

print("\n File tersimpan:")
print("   hasil/hasil_m1_m2.csv")
print("   hasil/eval_m1_m2.json")
print("   hasil/cm_vader_nltk.png")
print("   hasil/cm_textblob.png")
print("   hasil/perbandingan_vader_textblob.png")
print("\n Metode 1 & 2 selesai!")