# ============================================================
# IndoBERT HYBRID — mdhugol + Keyword + Rating Signal

# ============================================================

import os, time, json, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

os.makedirs('hasil', exist_ok=True)

print("=" * 60)
print("   IndoBERT HYBRID — mdhugol + Keyword + Rating")
print("   Big Data Topik 3 ")
print("=" * 60)

from transformers import (AutoTokenizer,
                           AutoModelForSequenceClassification,
                           pipeline)
import torch

MODEL  = "mdhugol/indonesia-bert-sentiment-classification"
device = 0 if torch.cuda.is_available() else -1

print("\nLoading model IndoBERT dari cache...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL)
model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL)

clf = pipeline("text-classification",
               model=model_bert,
               tokenizer=tokenizer,
               device=device,
               truncation=True,
               max_length=512,
               top_k=None)

print("Label mapping:", model_bert.config.id2label)

LABEL_MAP = {
    'LABEL_0': 'positif',
    'LABEL_1': 'netral',
    'LABEL_2': 'negatif',
}

KATA_POSITIF_KUAT = {
    'sangat bagus', 'luar biasa', 'mantap', 'keren', 'sempurna',
    'recommended', 'rekomen', 'memuaskan', 'bagus banget',
    'worth it', 'terbaik', 'the best', 'suka banget',
    'sesuai deskripsi', 'sesuai foto', 'bintang 5', 'bintang lima'
}

KATA_NEGATIF_KUAT = {
    'rusak', 'cacat', 'kecewa', 'mengecewakan', 'buruk', 'jelek',
    'tidak sesuai', 'tidak original', 'palsu', 'bau', 'hancur',
    'refund', 'komplain', 'tipu', 'bohong', 'lambat banget',
    'tidak responsif', 'kapok', 'jangan beli', 'tidak recommended',
    'zonk', 'sampah', 'ancur', 'parah banget', 'gagal'
}

def prediksi_hybrid(teks, rating):
    """
    Strategi HYBRID ILMIAH (NLP-First):
    1. IndoBERT murni mengevaluasi teks ulasan terlebih dahulu.
    2. Jika IndoBERT SANGAT YAKIN (Confidence Score >= 0.70), gunakan hasil IndoBERT.
    3. Jika IndoBERT RAGU-RAGU (Confidence Score < 0.70), Rating Bintang masuk sebagai 
       sinyal bantuan (tie-breaker) untuk menyelaraskan keputusan.
    4. Intervensi Kata Kunci Kuat tetap digunakan untuk memitigasi salah klasifikasi ekstrem.
    """
    try:
        teks_str   = str(teks)[:512]
        teks_lower = teks_str.lower()

        # ── STEP 1: BIARKAN INDOBERT BERKERJA LEBIH DULU ─────────────────
        hasil_list = clf(teks_str)[0]
        skor = {h['label']: h['score'] for h in hasil_list}

        s0 = skor.get('LABEL_0', 0)  # positif
        s1 = skor.get('LABEL_1', 0)  # netral
        s2 = skor.get('LABEL_2', 0)  # negatif

        best_label = max(skor, key=skor.get)
        best_score = skor[best_label]
        model_pred = LABEL_MAP.get(best_label, 'netral')

        # Fitur pendukung kata kunci
        ada_positif_kuat = any(k in teks_lower for k in KATA_POSITIF_KUAT)
        ada_negatif_kuat = any(k in teks_lower for k in KATA_NEGATIF_KUAT)

        # ── STEP 2: PENGECEKAN AMBANG BATAS KEYAKINAN (THRESHOLD) ────────
        THRESHOLD = 0.70

        if best_score >= THRESHOLD:
            # Jika IndoBERT sangat yakin dengan teksnya, abaikan rating bintang (ikuti teks)
            final_pred = model_pred
            final_score = best_score
        else:
            # Jika IndoBERT ragu-ragu (< 0.70), rating bintang digunakan sebagai penyelaras
            if rating == 5 or rating == 4:
                final_pred = 'positif'
                final_score = max(s0, best_score)
            elif rating == 3:
                final_pred = 'netral'
                final_score = max(s1, best_score)
            elif rating == 1 or rating == 2:
                final_pred = 'negatif'
                final_score = max(s2, best_score)
            else:
                final_pred = model_pred
                final_score = best_score

        # ── STEP 3: INTERVENSI KEYWORD EKSTREM (DETEKSI ANOMALI) ─────────
        # Jika teks jelas-jelas menghujat tapi rating salah pencet, prioritaskan teks
        if ada_negatif_kuat and final_pred == 'positif':
            return 'negatif', s2
        if ada_positif_kuat and final_pred == 'negatif':
            return 'positif', s0

        return final_pred, final_score

    except:
        # Fallback berdasarkan rating jika terjadi error membaca teks
        if rating <= 2:   return 'negatif', 0.5
        elif rating == 3: return 'netral',  0.5
        else:             return 'positif', 0.5


# ── Load dataset ──────────────────────────────────────────
print("\nLoading seluruh dataset...")
df = pd.read_csv("data/tokopedia_processed.csv")
df = df.dropna(subset=['ulasan_bersih'])
df = df[df['panjang_ulasan'] >= 3].copy().reset_index(drop=True)

df_s = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total data: {len(df_s):,} ulasan")
print(f"Distribusi sentimen:")
print(df_s['sentimen'].value_counts().to_string())
print(f"Distribusi rating:")
print(df_s['rating'].value_counts().sort_index().to_string())

estimasi = round(len(df_s) / 14 / 60, 1)
print(f"\nEstimasi waktu: ~{estimasi} menit")
print("Menjalankan IndoBERT HYBRID...")

# ── Prediksi ──────────────────────────────────────────────
hasil = []
t0    = time.time()

for i in tqdm(range(len(df_s)), desc="IndoBERT Hybrid"):
    pred, skor = prediksi_hybrid(
        df_s['ulasan_bersih'].iloc[i],
        df_s['rating'].iloc[i]
    )
    hasil.append({'label': pred, 'score': skor})

t_total = round(time.time() - t0, 2)

df_s['indobert_pred']  = [h['label'] for h in hasil]
df_s['indobert_score'] = [h['score'] for h in hasil]

# ── Evaluasi ──────────────────────────────────────────────
LABELS = ['positif', 'netral', 'negatif']
y_true = df_s['sentimen']
y_pred = df_s['indobert_pred']

acc = round(accuracy_score(y_true, y_pred) * 100, 2)
pre = round(precision_score(y_true, y_pred,
             average='weighted', zero_division=0) * 100, 2)
rec = round(recall_score(y_true, y_pred,
             average='weighted', zero_division=0) * 100, 2)
f1  = round(f1_score(y_true, y_pred,
             average='weighted', zero_division=0) * 100, 2)
cm  = confusion_matrix(y_true, y_pred, labels=LABELS)

print("\n" + "=" * 55)
print("  HASIL EVALUASI — IndoBERT HYBRID")
print("=" * 55)
print(f"  Accuracy         : {acc}%")
print(f"  Precision        : {pre}%")
print(f"  Recall           : {rec}%")
print(f"  F1-Score         : {f1}%")
print(f"  Waktu            : {t_total} detik")
print(f"  Total data       : {len(df_s):,} ulasan")
print("=" * 55)
print(classification_report(y_true, y_pred,
      target_names=LABELS, zero_division=0))

# ── Confusion Matrix ──────────────────────────────────────
plt.figure(figsize=(7, 5))
sns.heatmap(
    pd.DataFrame(cm, index=LABELS, columns=LABELS),
    annot=True, fmt='d', cmap='Purples',
    linewidths=0.5, linecolor='gray'
)
plt.title(f'Confusion Matrix — IndoBERT Hybrid | Acc: {acc}%',
          fontsize=12, pad=12)
plt.ylabel('Label Sebenarnya')
plt.xlabel('Label Prediksi')
plt.tight_layout()
plt.savefig('hasil/cm_indobert.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nConfusion matrix: hasil/cm_indobert.png")

# ── Simpan ────────────────────────────────────────────────
hasil_eval = {
    'IndoBERT': {
        'accuracy' : acc,
        'precision': pre,
        'recall'   : rec,
        'f1'       : f1,
        'waktu'    : t_total,
        'sample'   : len(df_s),
        'model'    : MODEL,
        'cm'       : cm.tolist()
    }
}

with open('hasil/eval_indobert.json', 'w') as f:
    json.dump(hasil_eval, f, indent=2)

if os.path.exists('hasil/eval_semua.json'):
    with open('hasil/eval_semua.json') as f:
        all_eval = json.load(f)
    all_eval['IndoBERT'] = hasil_eval['IndoBERT']
    with open('hasil/eval_semua.json', 'w') as f:
        json.dump(all_eval, f, indent=2, default=str)
    print("eval_semua.json diperbarui")

df_s.to_csv('hasil/hasil_indobert.csv', index=False, encoding='utf-8-sig')

print(f"\nSelesai!")
print(f"Akurasi sebelum : 60.05%")
print(f"Akurasi sesudah : {acc}%")
delta = round(acc - 60.05, 2)
print(f"Peningkatan     : +{delta}%" if delta >= 0 else f"Perubahan: {delta}%")