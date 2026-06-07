# ============================================================
# FLASK DASHBOARD — Sentiment Analysis Tokopedia
# Mata Kuliah : Big Data — Topik 3
# Cara jalankan:
#   1. pip install flask pandas nltk textblob scikit-learn joblib
#   2. python app.py
#   3. Buka: http://localhost:5000
# ============================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os, json, time, warnings, re
warnings.filterwarnings('ignore')

import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

app = Flask(__name__)

# ═══════════════════════════════════════════════════════
# LOAD SEMUA KOMPONEN SAAT STARTUP
# ═══════════════════════════════════════════════════════
sia = SentimentIntensityAnalyzer()

# Load model SVM dan RF
SVM_OK = False
RF_OK  = False
SP_OK  = False
try:
    import joblib
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    svm   = joblib.load("models/model_svm.pkl")
    SVM_OK = True
except: pass

try:
    import joblib
    rf = joblib.load("models/model_rf.pkl")
    RF_OK = True
except: pass

try:
    import joblib
    spark_pipe = joblib.load("models/spark_pipeline.pkl")
    SP_OK = True
except: pass

MODEL_OK = SVM_OK and RF_OK

# Load dataset
DF_OK = False
df_global = pd.DataFrame()
try:
    df_global = pd.read_csv("data/reporsed.csv")
    df_global = df_global.dropna(subset=['ulasan_bersih'])
    df_global = df_global[df_global['panjang_ulasan'] >= 3].copy()
    df_global = df_global.reset_index(drop=True)
    DF_OK = True
except Exception as e:
    print(f"Dataset error: {e}")

# Load hasil evaluasi dari JSON
EVAL_DATA = {}
for fname in ['eval_semua.json', 'eval_m1_m2.json', 'eval_m3.json',
              'eval_spark.json', 'eval_indobert.json']:
    fp = f'hasil/{fname}'
    if os.path.exists(fp):
        try:
            with open(fp, encoding='utf-8') as f:
                EVAL_DATA.update(json.load(f))
        except:
            pass

# Kalau tidak ada JSON, pakai data default dari hasil kamu
if not EVAL_DATA:
    EVAL_DATA = {
        "VADER"        : {"accuracy":34.32,"precision":46.31,"recall":34.32,"f1":24.83,"waktu":5.48},
        "TextBlob"     : {"accuracy":33.54,"precision":42.0, "recall":33.54,"f1":24.18,"waktu":8.79},
        "SVM"          : {"accuracy":70.69,"precision":70.17,"recall":70.69,"f1":70.31,"waktu_train":0.747},
        "Random_Forest": {"accuracy":69.14,"precision":68.91,"recall":69.14,"f1":67.74,"waktu_train":22.637},
        "Spark_MLlib"  : {"accuracy":62.61,"precision":62.53,"recall":62.61,"f1":62.54,"waktu_train":38.42},
        "IndoBERT"     : {"accuracy":58.61,"precision":54.92,"recall":58.61,"f1":52.29,"waktu":318.22},
    }

print("\n" + "="*55)
print("  Sentiment Analysis Tokopedia — Flask Dashboard")
print("  Big Data Topik 3")
print("="*55)
print(f"  Dataset  : {'OK — ' + str(len(df_global)) + ' ulasan' if DF_OK else 'TIDAK DITEMUKAN'}")
print(f"  SVM      : {'OK' if SVM_OK else 'Belum ada'}")
print(f"  RF       : {'OK' if RF_OK  else 'Belum ada'}")
print(f"  Spark    : {'OK' if SP_OK  else 'Belum ada (jalankan python -c membuat pkl)'}")
print(f"  Evaluasi : {list(EVAL_DATA.keys())}")
print("="*55)

# ═══════════════════════════════════════════════════════
# FUNGSI PREDIKSI
# ═══════════════════════════════════════════════════════
def pred_vader(teks):
    s = sia.polarity_scores(str(teks))
    c = s['compound']
    l = 'positif' if c >= 0.05 else ('negatif' if c <= -0.05 else 'netral')
    return {'label': l, 'compound': round(c,4),
            'pos': round(s['pos'],4), 'neg': round(s['neg'],4),
            'neu': round(s['neu'],4)}

def pred_textblob(teks):
    b = TextBlob(str(teks))
    p = b.sentiment.polarity
    s = b.sentiment.subjectivity
    l = 'positif' if p > 0.05 else ('negatif' if p < -0.05 else 'netral')
    return {'label': l, 'polarity': round(p,4), 'subjectivity': round(s,4)}

def pred_svm(teks):
    if not SVM_OK: return 'N/A'
    try:
        v = tfidf.transform([str(teks)])
        return svm.predict(v)[0]
    except: return 'N/A'

def pred_rf(teks):
    if not RF_OK: return 'N/A'
    try:
        v = tfidf.transform([str(teks)])
        return rf.predict(v)[0]
    except: return 'N/A'

def pred_spark(teks):
    if not SP_OK: return 'N/A'
    try:
        return spark_pipe.predict([str(teks)])[0]
    except: return 'N/A'

def voting(votes):
    valid = [v for v in votes if v != 'N/A']
    if not valid: return 'netral'
    return max(set(valid), key=valid.count)

# ═══════════════════════════════════════════════════════
# KONVERSI TANGGAL RELATIF
# ═══════════════════════════════════════════════════════
def parse_tanggal_relatif(teks):
    """Konversi '3 bulan lalu' ke nama bulan berdasarkan bulan ini"""
    import datetime
    now = datetime.datetime.now()
    t = str(teks).lower().strip()
    try:
        if 'lebih dari 1 tahun' in t or 'tahun' in t:
            d = now - datetime.timedelta(days=400)
        elif 'bulan lalu' in t:
            n = int(re.search(r'\d+', t).group())
            d = now - datetime.timedelta(days=n*30)
        elif 'minggu lalu' in t:
            n = int(re.search(r'\d+', t).group())
            d = now - datetime.timedelta(weeks=n)
        elif 'hari lalu' in t:
            n = int(re.search(r'\d+', t).group())
            d = now - datetime.timedelta(days=n)
        else:
            d = now
        return d.strftime('%Y-%m')
    except:
        return '2024-01'

# ═══════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════
@app.route('/')
def index():
    stats = {}
    if DF_OK:
        stats = {
            'total'   : len(df_global),
            'positif' : int(df_global['sentimen'].eq('positif').sum()),
            'negatif' : int(df_global['sentimen'].eq('negatif').sum()),
            'netral'  : int(df_global['sentimen'].eq('netral').sum()),
            'kategori': int(df_global['kategori'].nunique()),
            'produk'  : int(df_global['nama_produk'].nunique()),
        }
    return render_template('index.html',
        stats=stats,
        model_ok=MODEL_OK,
        svm_ok=SVM_OK,
        rf_ok=RF_OK,
        spark_ok=SP_OK)


@app.route('/api/analisis', methods=['POST'])
def analisis():
    data = request.get_json()
    teks = data.get('teks', '').strip()
    if not teks:
        return jsonify({'error': 'Teks kosong'}), 400

    t0  = time.time()
    v   = pred_vader(teks)
    tb  = pred_textblob(teks)
    s   = pred_svm(teks)
    r   = pred_rf(teks)
    sp  = pred_spark(teks)
    dur = round(time.time() - t0, 3)

    maj = voting([v['label'], tb['label'], s, r, sp])
    return jsonify({
        'teks'     : teks,
        'vader'    : v,
        'textblob' : tb,
        'svm'      : s,
        'rf'       : r,
        'spark'    : sp,
        'mayoritas': maj,
        'durasi'   : dur
    })


@app.route('/api/batch', methods=['POST'])
def batch():
    data      = request.get_json()
    teks_list = data.get('teks_list', [])
    if not teks_list:
        return jsonify({'error': 'List kosong'}), 400

    hasil = []
    dist  = {'positif': 0, 'negatif': 0, 'netral': 0}
    for teks in teks_list[:200]:
        v   = pred_vader(teks)
        tb  = pred_textblob(teks)
        s   = pred_svm(teks)
        r   = pred_rf(teks)
        sp  = pred_spark(teks)
        maj = voting([v['label'], tb['label'], s, r, sp])
        dist[maj] = dist.get(maj, 0) + 1
        hasil.append({
            'teks'    : str(teks)[:80],
            'vader'   : v['label'],
            'textblob': tb['label'],
            'svm'     : s,
            'rf'      : r,
            'spark'   : sp,
            'hasil'   : maj
        })
    return jsonify({'hasil': hasil, 'distribusi': dist, 'total': len(hasil)})


@app.route('/api/evaluasi')
def get_eval():
    return jsonify(EVAL_DATA)


@app.route('/api/distribusi_kategori')
def dist_kat():
    if not DF_OK: return jsonify({})
    res = {}
    for k in sorted(df_global['kategori'].unique()):
        sub = df_global[df_global['kategori'] == k]
        res[k] = {
            'positif': int(sub['sentimen'].eq('positif').sum()),
            'negatif': int(sub['sentimen'].eq('negatif').sum()),
            'netral' : int(sub['sentimen'].eq('netral').sum()),
            'total'  : len(sub)
        }
    return jsonify(res)


@app.route('/api/timeseries')
def timeseries():
    if not DF_OK: return jsonify([])
    try:
        d = df_global.copy()
        d['bulan'] = d['tanggal'].apply(parse_tanggal_relatif)
        ts = d.groupby(['bulan', 'sentimen']).size().unstack(fill_value=0)
        ts = ts.reset_index().sort_values('bulan')
        # Pastikan semua kolom ada
        for col in ['positif', 'negatif', 'netral']:
            if col not in ts.columns:
                ts[col] = 0
        return jsonify(ts[['bulan', 'positif', 'negatif', 'netral']].to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/processing_time')
def proc_time():
    hasil = {}
    for m, d in EVAL_DATA.items():
        t = d.get('waktu') or d.get('waktu_train') or 0
        hasil[m] = float(t)
    return jsonify(hasil)


@app.route('/api/sampel')
def sampel():
    if not DF_OK: return jsonify([])
    s = df_global.sample(min(50, len(df_global)), random_state=42)
    cols = ['nama_produk', 'kategori', 'rating', 'ulasan', 'sentimen', 'tanggal']
    return jsonify(s[cols].fillna('-').to_dict(orient='records'))


@app.route('/api/stats_kategori')
def stats_kat():
    if not DF_OK: return jsonify({})
    res = {}
    for k in sorted(df_global['kategori'].unique()):
        sub = df_global[df_global['kategori'] == k]
        pos = sub['sentimen'].eq('positif').sum()
        neg = sub['sentimen'].eq('negatif').sum()
        net = sub['sentimen'].eq('netral').sum()
        total = len(sub)
        res[k] = {
            'positif'  : int(pos),
            'negatif'  : int(neg),
            'netral'   : int(net),
            'total'    : total,
            'pct_pos'  : round(pos/total*100, 1) if total else 0,
            'avg_rating': round(sub['rating'].mean(), 2) if 'rating' in sub else 0,
        }
    return jsonify(res)


# ═══════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n  Buka browser: http://localhost:5000\n")
    app.run(debug=False, port=5000, host='0.0.0.0')
