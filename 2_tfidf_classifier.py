# ============================================================
# METODE 3 — TF-IDF + Classifier (SVM & Random Forest)
# ============================================================
# Mata Kuliah : Big Data — Topik 3
# Cara jalankan: python 2_tfidf_classifier.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib, time, json, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('hasil', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("   METODE 3 — TF-IDF + Classifier")
print("   SVM (LinearSVC) & Random Forest")
print("=" * 60)

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
df = pd.read_csv("data/tokopedia_processed.csv")
df = df.dropna(subset=['ulasan_bersih'])
df = df[df['panjang_ulasan'] >= 3].copy().reset_index(drop=True)

print(f"\n Dataset: {len(df):,} ulasan, {df['kategori'].nunique()} kategori")
print(f"   Distribusi sentimen:")
for s, n in df['sentimen'].value_counts().items():
    print(f"   {s:10s}: {n:,} ({n/len(df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────
# SPLIT DATA 80:20
# ──────────────────────────────────────────────────────────
X = df['ulasan_bersih']
y = df['sentimen']

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index,
    test_size=0.2, random_state=42, stratify=y
)

print(f"\n Split Data:")
print(f"   Train : {len(X_train):,} ulasan (80%)")
print(f"   Test  : {len(X_test):,} ulasan (20%)")

# ──────────────────────────────────────────────────────────
# TF-IDF VECTORIZER
# ──────────────────────────────────────────────────────────
print("\n Membuat TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

print(f" TF-IDF selesai")
print(f"   Jumlah fitur  : {X_train_vec.shape[1]:,}")
print(f"   Matrix train  : {X_train_vec.shape}")

# Simpan vectorizer
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print(f"   Tersimpan     : models/tfidf_vectorizer.pkl")

# ──────────────────────────────────────────────────────────
# FUNGSI EVALUASI
# ──────────────────────────────────────────────────────────
LABELS = ['positif', 'negatif', 'netral']

def evaluasi(y_true, y_pred, nama, t_train, t_pred):
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
    print(f"  Waktu Training   : {t_train} detik")
    print(f"  Waktu Prediksi   : {t_pred} detik")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))

    # Confusion Matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(pd.DataFrame(cm, index=LABELS, columns=LABELS),
                annot=True, fmt='d', cmap='Blues',
                linewidths=0.5, linecolor='gray')
    plt.title(f'Confusion Matrix — {nama}', fontsize=13, pad=12)
    plt.ylabel('Label Sebenarnya', fontsize=11)
    plt.xlabel('Label Prediksi', fontsize=11)
    plt.tight_layout()
    fn = nama.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    plt.savefig(f'hasil/cm_{fn}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrix: hasil/cm_{fn}.png")

    return dict(accuracy=acc, precision=pre, recall=rec, f1=f1,
                waktu_train=t_train, waktu_pred=t_pred, cm=cm.tolist())

# ──────────────────────────────────────────────────────────
# MODEL A — LINEAR SVM
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL A — Support Vector Machine (LinearSVC)")
print("=" * 60)
print("  Konfigurasi: kernel=linear, max_iter=2000")

t0 = time.time()
svm = LinearSVC(max_iter=2000, random_state=42, C=1.0)
svm.fit(X_train_vec, y_train)
t_train_svm = round(time.time() - t0, 3)

t0 = time.time()
y_pred_svm = svm.predict(X_test_vec)
t_pred_svm = round(time.time() - t0, 3)

eval_svm = evaluasi(y_test, y_pred_svm, 'SVM LinearSVC', t_train_svm, t_pred_svm)

# Cross Validation SVM
print("   Cross-Validation (5-fold)...")
cv_svm = cross_val_score(LinearSVC(max_iter=1000, random_state=42),
                          X_train_vec, y_train, cv=5, scoring='accuracy')
cv_mean = round(cv_svm.mean() * 100, 2)
cv_std  = round(cv_svm.std() * 100, 2)
print(f"  CV Accuracy: {cv_mean}% ± {cv_std}%")
eval_svm['cv_mean'] = cv_mean
eval_svm['cv_std']  = cv_std

joblib.dump(svm, 'models/model_svm.pkl')
print("   Model disimpan: models/model_svm.pkl")

# ──────────────────────────────────────────────────────────
# MODEL B — RANDOM FOREST
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL B — Random Forest")
print("=" * 60)
print("  Konfigurasi: n_estimators=200, n_jobs=-1")

t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_vec, y_train)
t_train_rf = round(time.time() - t0, 3)

t0 = time.time()
y_pred_rf = rf.predict(X_test_vec)
t_pred_rf = round(time.time() - t0, 3)

eval_rf = evaluasi(y_test, y_pred_rf, 'Random Forest', t_train_rf, t_pred_rf)

print("   Cross-Validation RF (5-fold)...")
cv_rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                         X_train_vec, y_train, cv=5, scoring='accuracy')
eval_rf['cv_mean'] = round(cv_rf.mean() * 100, 2)
eval_rf['cv_std']  = round(cv_rf.std() * 100, 2)
print(f"  CV Accuracy: {eval_rf['cv_mean']}% ± {eval_rf['cv_std']}%")

joblib.dump(rf, 'models/model_rf.pkl')
print("   Model disimpan: models/model_rf.pkl")

# ──────────────────────────────────────────────────────────
# PERBANDINGAN SVM vs RF
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PERBANDINGAN SVM vs RANDOM FOREST")
print("=" * 60)

tbl = pd.DataFrame({
    'Metode'      : ['SVM (LinearSVC)', 'Random Forest'],
    'Accuracy (%)'  : [eval_svm['accuracy'], eval_rf['accuracy']],
    'Precision (%)' : [eval_svm['precision'], eval_rf['precision']],
    'Recall (%)'    : [eval_svm['recall'],    eval_rf['recall']],
    'F1-Score (%)'  : [eval_svm['f1'],        eval_rf['f1']],
    'CV Acc (%)'    : [f"{eval_svm['cv_mean']}±{eval_svm['cv_std']}",
                       f"{eval_rf['cv_mean']}±{eval_rf['cv_std']}"],
    'Train (s)'   : [eval_svm['waktu_train'], eval_rf['waktu_train']],
})
print(tbl.to_string(index=False))

# Grafik perbandingan
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrik = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
sv = [eval_svm['accuracy'], eval_svm['precision'], eval_svm['recall'], eval_svm['f1']]
rv = [eval_rf['accuracy'],  eval_rf['precision'],  eval_rf['recall'],  eval_rf['f1']]
x = np.arange(4); w = 0.35
b1 = axes[0].bar(x-w/2, sv, w, label='SVM', color='#8E44AD', alpha=0.85)
b2 = axes[0].bar(x+w/2, rv, w, label='RF',  color='#E67E22', alpha=0.85)
axes[0].set_title('Metrik Evaluasi', fontsize=12)
axes[0].set_xticks(x); axes[0].set_xticklabels(metrik, fontsize=9)
axes[0].set_ylim(0, 100); axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
axes[0].bar_label(b1, fmt='%.1f', padding=2, fontsize=8)
axes[0].bar_label(b2, fmt='%.1f', padding=2, fontsize=8)

# Waktu training
wb = axes[1].bar(['SVM','RF'], [eval_svm['waktu_train'], eval_rf['waktu_train']],
                  color=['#8E44AD','#E67E22'], alpha=0.85, width=0.4)
axes[1].set_title('Waktu Training (detik)', fontsize=12)
axes[1].set_ylabel('Detik'); axes[1].grid(axis='y', alpha=0.3)
axes[1].bar_label(wb, fmt='%.2fs', padding=3, fontsize=9)

# Akurasi per kategori
df_test_kat = df.iloc[idx_test].copy()
df_test_kat['pred_svm'] = y_pred_svm
df_test_kat['pred_rf']  = y_pred_rf
per_kat = df_test_kat.groupby('kategori').apply(
    lambda g: pd.Series({
        'SVM': round(accuracy_score(g['sentimen'], g['pred_svm'])*100, 1),
        'RF' : round(accuracy_score(g['sentimen'], g['pred_rf'])*100, 1),
    })
).sort_values('SVM', ascending=True)
per_kat.plot(kind='barh', ax=axes[2], color=['#8E44AD','#E67E22'],
             alpha=0.85, width=0.7)
axes[2].set_title('Akurasi per Kategori', fontsize=12)
axes[2].set_xlabel('Akurasi (%)')
axes[2].grid(axis='x', alpha=0.3)
axes[2].set_xlim(0, 100)

plt.tight_layout()
plt.savefig('hasil/perbandingan_svm_rf.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   Grafik disimpan: hasil/perbandingan_svm_rf.png")

# Akurasi per kategori detail
print("\n   Akurasi per Kategori:")
print(f"  {'Kategori':<15} {'SVM':>8} {'RF':>8} {'Jumlah':>8}")
print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8}")
for kat in df_test_kat['kategori'].unique():
    sub = df_test_kat[df_test_kat['kategori']==kat]
    a_svm = round(accuracy_score(sub['sentimen'], sub['pred_svm'])*100, 1)
    a_rf  = round(accuracy_score(sub['sentimen'], sub['pred_rf'])*100, 1)
    print(f"  {kat:<15} {a_svm:>7.1f}% {a_rf:>7.1f}% {len(sub):>8,}")

# ──────────────────────────────────────────────────────────
# SIMPAN HASIL
# ──────────────────────────────────────────────────────────
df_test_kat.to_csv('hasil/hasil_m3.csv', index=False, encoding='utf-8-sig')
with open('hasil/eval_m3.json', 'w') as f:
    json.dump({'SVM': eval_svm, 'Random_Forest': eval_rf}, f, indent=2, default=str)

print("\n File tersimpan:")
print("   hasil/hasil_m3.csv")
print("   hasil/eval_m3.json")
print("   hasil/cm_svm_linearSVC.png")
print("   hasil/cm_random_forest.png")
print("   hasil/perbandingan_svm_rf.png")
print("   models/model_svm.pkl")
print("   models/model_rf.pkl")
print("   models/tfidf_vectorizer.pkl")
print("\n Metode 3 selesai!")