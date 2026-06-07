============================================================
  SENTIMENT ANALYSIS TOKOPEDIA — Big Data Topik 3
  Universitas Pignateli Triputra
============================================================

STRUKTUR FOLDER:
================
tokopedia/
├── app.py                    <- Flask Dashboard (jalankan ini)
├── requirements.txt          <- Daftar library
├── 1_vader_textblob.py       <- Metode 1 & 2
├── 2_tfidf_classifier.py     <- Metode 3 (SVM + RF)
├── 3_sparkmllib.py           <- Metode 4 (Spark MLlib)
├── 4_indobert.py             <- Bonus IndoBERT
├── tokopedia2_scraping.py    <- Script scraping data
├── tokopediaprocessed.py     <- Script preprocessing
├── data/
│   ├── tokopedia_baru.csv    <- Data mentah
│   └── tokopedia_processed.csv  <- Data terproses
├── hasil/                    <- Hasil evaluasi & grafik
├── models/                   <- Model yang sudah dilatih
│   ├── model_svm.pkl
│   ├── model_rf.pkl
│   ├── tfidf_vectorizer.pkl
│   └── spark_pipeline.pkl
└── templates/
    └── index.html            <- UI Flask Dashboard

CARA JALANKAN FLASK DASHBOARD:
================================
1. Install library:
   pip install flask pandas numpy scikit-learn joblib nltk textblob

2. Jalankan Flask:
   python app.py

3. Buka browser:
   http://localhost:5000

FITUR DASHBOARD:
================
- Analisis Real-time : Input teks, hasil 5 metode + voting
- Batch Analysis     : Analisis banyak ulasan sekaligus
- Evaluasi Model     : Tabel & grafik perbandingan semua metode
- Visualisasi Data   : Pie, bar kategori, time-series
- Processing Time    : Perbandingan waktu setiap metode
- Data Sampel        : Tabel 50 baris data

HASIL EVALUASI:
===============
Metode          Accuracy   F1-Score   Waktu
--------------  ---------  ---------  ------
VADER           34.32%     24.83%     5.48s
TextBlob        33.54%     24.18%     8.79s
SVM             70.69%     70.31%     0.75s   <- Terbaik non-DL
Random Forest   69.14%     67.74%     22.6s
Spark MLlib     62.61%     62.54%     38.4s
IndoBERT        58.61%     52.29%     318s
============================================================
