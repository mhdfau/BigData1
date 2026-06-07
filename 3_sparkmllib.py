# ============================================================
# METODE 4 — Spark MLlib Pipeline
# ============================================================
# Mata Kuliah : Big Data — Topik 3
# Pipeline    : Tokenizer → HashingTF → IDF → Classifier
# Cara jalankan: python 3_spark_mllib.py
# ============================================================
import sys
import os
import shutil
from datetime import datetime
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['PYTHONUNBUFFERED'] = '1'



import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time, json, warnings
warnings.filterwarnings('ignore')

os.makedirs('hasil', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 60)
print("   METODE SPARK — Spark MLlib Pipeline")
print("   Tokenizer → HashingTF → IDF → Classifier")
print("   Big Data Topik 3 ")
print("=" * 60)
print(f"Python yang dipakai: {sys.executable}") 

# ──────────────────────────────────────────────────────────
# CEK PYSPARK
# ──────────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (Tokenizer, HashingTF,
                                     IDF, StringIndexer)
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    SPARK_OK = True
except ImportError:
    SPARK_OK = False
    print("PySpark tidak tersedia!")
    print("Install: pip install pyspark")
    exit(1)

# ──────────────────────────────────────────────────────────
# SPARK SESSION
# ──────────────────────────────────────────────────────────
print("\n Menginisialisasi Spark Session...")
spark = SparkSession.builder \
    .appName("BigData_Topik3_SentimentTokopedia") \
    .master("local[1]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.extraJavaOptions", "-Xss4m") \
    .config("spark.python.worker.reuse", "false") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"Spark aktif — versi {spark.version}")
print(f"   Mode: local[*] (semua CPU core)")

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
df_pd = pd.read_csv("data/tokopedia_processed.csv")
df_pd = df_pd.dropna(subset=['ulasan_bersih'])
df_pd = df_pd[df_pd['panjang_ulasan'] >= 3].copy()
df_pd = df_pd[['ulasan_bersih', 'sentimen', 'kategori']].reset_index(drop=True)

df_spark = spark.createDataFrame(df_pd)
total    = df_spark.count()

print(f"\nData dimuat ke Spark: {total:,} baris")
print("   Schema:")
df_spark.printSchema()
print("   Contoh data:")
df_spark.show(3, truncate=60)

# Distribusi per kelas
print("   Distribusi sentimen:")
df_spark.groupBy('sentimen').count().show()

# ──────────────────────────────────────────────────────────
# SPLIT DATA
# ──────────────────────────────────────────────────────────
train_df, test_df = df_spark.randomSplit([0.8, 0.2], seed=42)
n_train = train_df.count()
n_test  = test_df.count()
print(f"Split data: Train={n_train:,} | Test={n_test:,}")

# ──────────────────────────────────────────────────────────
# BANGUN PIPELINE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MEMBANGUN SPARK MLLIB PIPELINE")
print("=" * 60)
print("""
  Alur Pipeline:
  ┌─────────────────────────────────────────┐
  │  ulasan_bersih (teks)                   │
  │       ↓ StringIndexer                   │
  │       ↓ Tokenizer  (kalimat → kata)     │
  │       ↓ HashingTF  (kata → hash vector) │
  │       ↓ IDF        (bobot TF-IDF)       │
  │       ↓ LogisticRegression              │
  │  prediksi sentimen                      │
  └─────────────────────────────────────────┘
""")

# Komponen pipeline
label_idx = StringIndexer(
    inputCol="sentimen",
    outputCol="label",
    handleInvalid="keep"
)
tokenizer = Tokenizer(
    inputCol="ulasan_bersih",
    outputCol="words"
)
hashTF = HashingTF(
    inputCol="words",
    outputCol="rawFeatures",
    numFeatures=10000
)
idf = IDF(
    inputCol="rawFeatures",
    outputCol="features",
    minDocFreq=2
)
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)

pipeline = Pipeline(stages=[label_idx, tokenizer, hashTF, idf, lr])

# ──────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────
print(" Training Spark MLlib Pipeline...")
t0 = time.time()
model = pipeline.fit(train_df)
t_train = round(time.time() - t0, 2)
print(f"Training selesai dalam {t_train} detik")

# ──────────────────────────────────────────────────────────
# PREDIKSI
# ──────────────────────────────────────────────────────────
print(" Prediksi pada data test...")
t0 = time.time()
prediksi = model.transform(test_df)
t_pred   = round(time.time() - t0, 2)
print(f"Prediksi selesai dalam {t_pred} detik")

print("\n  Contoh hasil prediksi:")
prediksi.select("ulasan_bersih", "sentimen", "label",
                "prediction").show(8, truncate=50)

# ──────────────────────────────────────────────────────────
# EVALUASI
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUASI SPARK MLLIB PIPELINE")
print("=" * 60)

def spark_metrik(metric):
    ev = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName=metric)
    return round(ev.evaluate(prediksi) * 100, 2)

acc = spark_metrik("accuracy")
f1  = spark_metrik("f1")
pre = spark_metrik("weightedPrecision")
rec = spark_metrik("weightedRecall")

print(f"  Accuracy         : {acc}%")
print(f"  Precision        : {pre}%")
print(f"  Recall           : {rec}%")
print(f"  F1-Score         : {f1}%")
print(f"  Waktu Training   : {t_train} detik")
print(f"  Waktu Prediksi   : {t_pred} detik")
print(f"  Total data       : {total:,} ulasan")

# Confusion matrix (konversi ke pandas)
from sklearn.metrics import confusion_matrix as sk_cm, classification_report as sk_cr

pred_pd    = prediksi.select("sentimen","label","prediction").toPandas()
label_list = model.stages[0].labels
pred_pd['pred_label'] = pred_pd['prediction'].apply(
    lambda x: label_list[int(x)] if int(x) < len(label_list) else 'netral')

LABELS = ['positif','negatif','netral']
labs   = [l for l in LABELS if l in pred_pd['sentimen'].unique()]
cm     = sk_cm(pred_pd['sentimen'], pred_pd['pred_label'], labels=labs)

print(f"\n  Classification Report:")
print(sk_cr(pred_pd['sentimen'], pred_pd['pred_label'],
            target_names=labs, zero_division=0))

plt.figure(figsize=(7, 5))
sns.heatmap(pd.DataFrame(cm, index=labs, columns=labs),
            annot=True, fmt='d', cmap='Reds',
            linewidths=0.5, linecolor='gray')
plt.title('Confusion Matrix — Spark MLlib Pipeline', fontsize=13, pad=12)
plt.ylabel('Label Sebenarnya', fontsize=11)
plt.xlabel('Label Prediksi', fontsize=11)
plt.tight_layout()
plt.savefig('hasil/cm_spark_mllib.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Confusion matrix: hasil/cm_spark_mllib.png")

# ──────────────────────────────────────────────────────────
# VISUALISASI PROCESSING TIME
# ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Metrik bar
metrik_names = ['Accuracy','Precision','Recall','F1-Score']
vals = [acc, pre, rec, f1]
bars = ax1.bar(metrik_names, vals, color=['#C0392B','#E74C3C','#EC7063','#F1948A'],
               alpha=0.9, width=0.5)
ax1.set_title('Metrik Evaluasi Spark MLlib', fontsize=12)
ax1.set_ylim(0, 100); ax1.set_ylabel('Nilai (%)')
ax1.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Pipeline alur visualisasi
steps = ['StringIndexer','Tokenizer','HashingTF','IDF','LogisticReg']
colors = ['#3498DB','#27AE60','#F39C12','#E74C3C','#8E44AD']
for i, (s, c) in enumerate(zip(steps, colors)):
    ax2.barh(i, 1, color=c, alpha=0.85, height=0.6)
    ax2.text(0.5, i, s, ha='center', va='center',
             color='white', fontweight='bold', fontsize=10)
    if i < len(steps)-1:
        ax2.annotate('', xy=(0.5, i-0.25), xytext=(0.5, i-0.6+0.25),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax2.set_title('Spark MLlib Pipeline Stages', fontsize=12)
ax2.set_xlim(0, 1); ax2.set_yticks([])
ax2.set_xticks([]); ax2.spines[['top','right','bottom','left']].set_visible(False)

plt.tight_layout()
plt.savefig('hasil/spark_pipeline_visual.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Visualisasi pipeline: hasil/spark_pipeline_visual.png")

# ──────────────────────────────────────────────────────────
# SIMPAN MODEL & HASIL
# ──────────────────────────────────────────────────────────
MODEL_PATH = "models/spark_pipeline_model"
try:
    if os.path.exists(MODEL_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{MODEL_PATH}_backup_{timestamp}"
        shutil.move(MODEL_PATH, backup_path)
        print(f" Folder lama direname ke: {backup_path}")
    model.write().save(MODEL_PATH)
    print(f" Model Spark tersimpan: {MODEL_PATH}/")
except Exception as e:
    print(f"  Model Spark tidak tersimpan: {e}")

pred_pd.to_csv('hasil/hasil_spark.csv', index=False, encoding='utf-8-sig')

eval_spark = dict(accuracy=acc, precision=pre, recall=rec, f1=f1,
                  waktu_train=t_train, waktu_pred=t_pred, cm=cm.tolist())
with open('hasil/eval_spark.json', 'w') as f:
    json.dump({'Spark_MLlib': eval_spark}, f, indent=2, default=str)

spark.stop()
print("\nSpark Session ditutup")
print("\nFile tersimpan:")
print("   hasil/hasil_spark.csv")
print("   hasil/eval_spark.json")
print("   hasil/cm_spark_mllib.png")
print("   hasil/spark_pipeline_visual.png")
print("\nSpark MLlib Pipeline selesai!")