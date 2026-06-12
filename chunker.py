"""
NLP Projesi: Isim ve Diger Obeklerin Saptanmasi (Chunking)
==========================================================
Bursa Teknik Universitesi - 2025-2026 Bahar Donemi

Bu dosya:
  - CoNLL formatindaki veriyi okur
  - Oznitelikler cikarir (kelime, suffix, komsular, vb.)
  - CRF ve MLP modelleri egitir
  - Sonuclari degerlendirir (F1, Precision, Recall, Accuracy)
  - Confusion matrix ve metrik grafigi olusturur

Kullanim:
  python chunker.py

Ciktilar:
  - report/confusion_matrix_crf.png
  - report/confusion_matrix_mlp.png
  - report/metrics_comparison.png
  - report/predictions_crf.conll
  - Terminal: her sinif icin metrik tablosu
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Sunucu/headless ortam için
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

# =====================================================
# 1. VERİ OKUMA
# =====================================================

def read_conll(filepath):
    """
    3 sutunlu CoNLL formatini okur.
    Dondurur: [(kelime_listesi, chunk_etiket_listesi, clause_etiket_listesi), ...]
    Format: ID FORM CHUNK-OUTER CHUNK-INNER CLAUSE
    """
    sentences = []
    words, chunk_labels, clause_labels = [], [], []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                if words:
                    sentences.append((words[:], chunk_labels[:], clause_labels[:]))
                    words, chunk_labels, clause_labels = [], [], []
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                words.append(parts[1])        # FORM
                chunk_labels.append(parts[2]) # CHUNK-OUTER
                # CLAUSE: 5. sutun varsa oku, yoksa "O"
                clause_labels.append(parts[4] if len(parts) >= 5 else "O")

    if words:
        sentences.append((words, chunk_labels, clause_labels))

    return sentences


# =====================================================
# 2. ÖZNİTELİK ÇIKARIMI (Feature Engineering)
# =====================================================

def word_features(sent, i):
    """
    Verilen cümledeki i. kelime için öznitelik sözlüğü döndürür.
    CRF modeli için kullanılır.
    """
    word = sent[i]

    features = {
        # --- Kelime özellikleri ---
        "word":         word,
        "word.lower":   word.lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.len":     len(word),

        # --- Suffix (son ekler - Türkçe için kritik!) ---
        "word[-1:]":    word[-1:],
        "word[-2:]":    word[-2:],
        "word[-3:]":    word[-3:],
        "word[-4:]":    word[-4:],

        # --- Prefix ---
        "word[:2]":     word[:2],
        "word[:3]":     word[:3],

        # --- Cümle pozisyonu ---
        "position":     i,
    }

    # --- Önceki kelime (i-1) ---
    if i > 0:
        prev = sent[i - 1]
        features.update({
            "-1:word":         prev,
            "-1:word.lower":   prev.lower(),
            "-1:word.isupper": prev.isupper(),
            "-1:word.istitle": prev.istitle(),
            "-1:word[-2:]":    prev[-2:],
            "-1:word[-3:]":    prev[-3:],
        })
    else:
        features["BOS"] = True  # Cümle başı (Beginning Of Sentence)

    # --- İki önceki kelime (i-2) ---
    if i > 1:
        prev2 = sent[i - 2]
        features.update({
            "-2:word":       prev2,
            "-2:word.lower": prev2.lower(),
            "-2:word[-2:]":  prev2[-2:],
        })

    # --- Sonraki kelime (i+1) ---
    if i < len(sent) - 1:
        nxt = sent[i + 1]
        features.update({
            "+1:word":         nxt,
            "+1:word.lower":   nxt.lower(),
            "+1:word.isupper": nxt.isupper(),
            "+1:word.istitle": nxt.istitle(),
            "+1:word[-2:]":    nxt[-2:],
            "+1:word[-3:]":    nxt[-3:],
        })
    else:
        features["EOS"] = True  # Cümle sonu (End Of Sentence)

    # --- İki sonraki kelime (i+2) ---
    if i < len(sent) - 2:
        nxt2 = sent[i + 2]
        features.update({
            "+2:word":       nxt2,
            "+2:word.lower": nxt2.lower(),
            "+2:word[-2:]":  nxt2[-2:],
        })

    return features


def sent_to_features(sent):
    """Cümlenin tüm kelimelerini öznitelik listesine çevirir (CRF için)."""
    return [word_features(sent, i) for i in range(len(sent))]


def sent_to_labels(labels):
    """Etiket listesini döndürür."""
    return labels


# =====================================================
# MLP İÇİN: Düzleştirilmiş vektör öznitelikleri
# =====================================================

def build_vocab(sentences, min_freq=1):
    """Kelime sozlugu olusturur."""
    from collections import Counter
    word_counts = Counter()
    for words, *_ in sentences:
        word_counts.update(w.lower() for w in words)
    vocab = {"<UNK>": 0, "<PAD>": 1}
    for word, cnt in word_counts.items():
        if cnt >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def word_to_vector(word, vocab, window_words, suffix_len=3):
    """Tek kelime için sayısal vektör oluşturur (MLP için)."""
    features = []

    # Kelime ID'si (one-hot değil, index - embedding gibi)
    idx = vocab.get(word.lower(), 0)
    features.append(idx / max(len(vocab), 1))  # Normalize

    # İkili özellikler
    features.append(1.0 if word.isupper() else 0.0)
    features.append(1.0 if word.istitle() else 0.0)
    features.append(1.0 if word.isdigit() else 0.0)
    features.append(min(len(word), 20) / 20.0)  # Uzunluk, normalize

    # Suffix hash (son 2 ve 3 karakter)
    for suf in [word[-2:], word[-3:]]:
        h = hash(suf) % 1000
        features.append(h / 1000.0)

    # Pencere kelimeleri (önceki ve sonraki)
    for w in window_words:
        w_idx = vocab.get(w.lower(), 0) if w else 0
        features.append(w_idx / max(len(vocab), 1))

    return features


def build_mlp_features(sentences, vocab, window=2):
    """Tum veri setini NB icin X, y dizilerine cevirir."""
    X, y = [], []
    for words, labels, *_ in sentences:
        for i, (word, label) in enumerate(zip(words, labels)):
            window_words = []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                j = i + offset
                window_words.append(words[j] if 0 <= j < len(words) else "")

            vec = word_to_vector(word, vocab, window_words)
            X.append(vec)
            y.append(label)

    return np.array(X, dtype=np.float64), np.array(y)


# =====================================================
# 3. GÖRSELLEŞTİRME
# =====================================================

os.makedirs("report", exist_ok=True)

# Renk paleti
COLORS = {
    "B-NP":   "#4F86C6",
    "I-NP":   "#7BAFD4",
    "B-VP":   "#E8734A",
    "I-VP":   "#F0A07A",
    "B-ADVP": "#6AAB6E",
    "I-ADVP": "#96CC99",
    "B-PP":   "#C07AB8",
    "I-PP":   "#D4A0CC",
    "O":      "#AAAAAA",
}


def plot_confusion_matrix(cm, labels, title, filepath):
    """Confusion matrix ısı haritasını çizer ve kaydeder."""
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(7, len(labels) - 1)))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"shrink": 0.8}
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Tahmin Edilen Etiket", fontsize=11)
    ax.set_ylabel("Gerçek Etiket", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion matrix kaydedildi: {filepath}")


def plot_metrics(report_dict, model_name, filepath):
    """Her sınıf için Precision, Recall, F1 bar chart çizer."""
    # Sınıf satırlarını filtrele (avg satırlarını çıkar)
    classes = [k for k in report_dict.keys()
               if k not in ("accuracy", "macro avg", "weighted avg")]

    precs   = [report_dict[c]["precision"]    for c in classes]
    recs    = [report_dict[c]["recall"]       for c in classes]
    f1s     = [report_dict[c]["f1-score"]     for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.2), 6))

    bars_p = ax.bar(x - w,   precs, w, label="Precision", color="#4F86C6", alpha=0.85)
    bars_r = ax.bar(x,       recs,  w, label="Recall",    color="#E8734A", alpha=0.85)
    bars_f = ax.bar(x + w,   f1s,   w, label="F1-Score",  color="#6AAB6E", alpha=0.85)

    # Değerleri bar üstüne yaz
    for bar in [*bars_p, *bars_r, *bars_f]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_title(f"{model_name} — Sınıf Bazlı Metrikler\n(Precision / Recall / F1-Score)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Metrik grafiği kaydedildi: {filepath}")


def plot_model_comparison(results, filepath):
    """İki modelin karşılaştırma grafiğini çizer."""
    models   = list(results.keys())
    metrics  = ["Accuracy", "Macro F1", "Weighted F1"]
    colors   = ["#4F86C6", "#E8734A"]

    vals = {
        m: [
            results[m]["accuracy"],
            results[m]["macro_f1"],
            results[m]["weighted_f1"],
        ]
        for m in models
    }

    x = np.arange(len(metrics))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (model, color) in enumerate(zip(models, colors)):
        offset = (idx - 0.5) * w
        bars = ax.bar(x + offset, vals[model], w, label=model, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_title("CRF vs MLP — Model Karşılaştırması", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Karşılaştırma grafiği kaydedildi: {filepath}")


def save_predictions_conll(sentences, all_preds, filepath):
    """Model tahminlerini 5 sutunlu CoNLL formatinda kaydeder."""
    with open(filepath, "w", encoding="utf-8") as f:
        for sent_tuple, pred_labels in zip(sentences, all_preds):
            words      = sent_tuple[0]
            true_chunk = sent_tuple[1]
            true_clause = sent_tuple[2] if len(sent_tuple) > 2 else ["O"] * len(words)

            text = " ".join(words)
            f.write(f"# text = {text}\n")
            f.write(f"# columns = ID FORM CHUNK-TRUE CHUNK-PRED CLAUSE\n")
            for i, (word, true_c, pred_c, clause) in enumerate(
                    zip(words, true_chunk, pred_labels, true_clause), 1):
                f.write(f"{i}\t{word}\t{true_c}\t{pred_c}\t{clause}\n")
            f.write("\n")
    print(f"  OK - Tahminler kaydedildi: {filepath}")



# =====================================================
# 4. ANA EĞİTİM VE DEĞERLENDİRME
# =====================================================

def train_and_evaluate():
    print("\n" + "=" * 60)
    print("NLP Chunking Projesi — Model Eğitimi ve Değerlendirmesi")
    print("=" * 60)

    # --- Veri yükle ---
    print("\n[1/5] Veri yükleniyor...")
    if not os.path.exists("data/train.conll") or not os.path.exists("data/test.conll"):
        print("  HATA: data/train.conll veya data/test.conll bulunamadı!")
        print("  Lütfen önce 'python prepare_data.py' komutunu çalıştırın.")
        sys.exit(1)

    train_sents = read_conll("data/train.conll")
    test_sents  = read_conll("data/test.conll")
    print(f"  Egitim: {len(train_sents)} cumle, {sum(len(w) for w, _, _ in train_sents)} token")
    print(f"  Test:   {len(test_sents)} cumle, {sum(len(w) for w, _, _ in test_sents)} token")

    # Tum sinif etiketlerini topla (CHUNK-OUTER)
    all_labels_train = [label for _, labels, _ in train_sents for label in labels]
    all_labels_test  = [label for _, labels, _ in test_sents  for label in labels]
    unique_labels = sorted(set(all_labels_train) | set(all_labels_test))
    print(f"  Siniflar ({len(unique_labels)}): {', '.join(unique_labels)}")

    results = {}

    # ===========================
    # MODEL 1: CRF
    # ===========================
    print("\n[2/5] CRF Modeli egitiliyor...")

    X_train_crf = [sent_to_features(words) for words, _, _ in train_sents]
    y_train_crf = [sent_to_labels(labels)  for _, labels, _ in train_sents]
    X_test_crf  = [sent_to_features(words) for words, _, _ in test_sents]
    y_test_crf  = [sent_to_labels(labels)  for _, labels, _ in test_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.05,
        c2=0.05,
        max_iterations=150,
        all_possible_transitions=True
    )
    crf.fit(X_train_crf, y_train_crf)
    print("  ✓ CRF eğitimi tamamlandı.")

    y_pred_crf = crf.predict(X_test_crf)

    # Düzleştir
    y_true_flat_crf = [l for labels in y_test_crf  for l in labels]
    y_pred_flat_crf = [l for labels in y_pred_crf  for l in labels]

    # CRF metrikleri
    crf_acc    = accuracy_score(y_true_flat_crf, y_pred_flat_crf)
    crf_macro  = f1_score(y_true_flat_crf, y_pred_flat_crf, average="macro",    zero_division=0)
    crf_weight = f1_score(y_true_flat_crf, y_pred_flat_crf, average="weighted", zero_division=0)

    from sklearn.metrics import classification_report as cr
    crf_report_str  = cr(y_true_flat_crf, y_pred_flat_crf,
                         labels=unique_labels, zero_division=0, digits=4)
    crf_report_dict = cr(y_true_flat_crf, y_pred_flat_crf,
                         labels=unique_labels, zero_division=0, output_dict=True)

    results["CRF"] = {
        "accuracy":    crf_acc,
        "macro_f1":    crf_macro,
        "weighted_f1": crf_weight,
        "report_dict": crf_report_dict,
    }

    print(f"\n  CRF Sonuclari:")
    print(f"    Accuracy   : {crf_acc:.4f}")
    print(f"    Macro F1   : {crf_macro:.4f}")
    print(f"    Weighted F1: {crf_weight:.4f}")
    print(f"\n  Sinif Bazli Rapor (CRF):")
    print(crf_report_str)

    # ===========================
    # MODEL 2: Naive Bayes
    # ===========================
    print("\n[3/5] Naive Bayes Modeli egitiliyor...")

    vocab = build_vocab(train_sents, min_freq=1)
    X_train_nb, y_train_nb_str = build_mlp_features(train_sents, vocab)
    X_test_nb,  y_test_nb_str  = build_mlp_features(test_sents,  vocab)

    # MinMaxScaler: Naive Bayes negatif deger kabul etmez
    scaler = MinMaxScaler()
    X_train_nb = scaler.fit_transform(X_train_nb)
    X_test_nb  = scaler.transform(X_test_nb)

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_nb, y_train_nb_str)
    print("  OK - Naive Bayes egitimi tamamlandi.")

    y_pred_nb = nb.predict(X_test_nb)

    nb_acc    = accuracy_score(y_test_nb_str, y_pred_nb)
    nb_macro  = f1_score(y_test_nb_str, y_pred_nb, average="macro",    zero_division=0)
    nb_weight = f1_score(y_test_nb_str, y_pred_nb, average="weighted", zero_division=0)

    nb_report_str  = cr(y_test_nb_str, y_pred_nb,
                        labels=unique_labels, zero_division=0, digits=4)
    nb_report_dict = cr(y_test_nb_str, y_pred_nb,
                        labels=unique_labels, zero_division=0, output_dict=True)

    results["Naive Bayes"] = {
        "accuracy":    nb_acc,
        "macro_f1":    nb_macro,
        "weighted_f1": nb_weight,
        "report_dict": nb_report_dict,
    }

    print(f"\n  Naive Bayes Sonuclari:")
    print(f"    Accuracy   : {nb_acc:.4f}")
    print(f"    Macro F1   : {nb_macro:.4f}")
    print(f"    Weighted F1: {nb_weight:.4f}")
    print(f"\n  Sinif Bazli Rapor (Naive Bayes):")
    print(nb_report_str)

    # ===========================
    # 4. GRAFİKLER
    # ===========================
    print("\n[4/5] Grafikler olusturuluyor...")

    # CRF Confusion Matrix
    cm_crf = confusion_matrix(y_true_flat_crf, y_pred_flat_crf, labels=unique_labels)
    plot_confusion_matrix(cm_crf, unique_labels,
                          "CRF — Karmaşıklık Matrisi (Confusion Matrix)",
                          "report/confusion_matrix_crf.png")

    # NB Confusion Matrix
    cm_nb = confusion_matrix(y_test_nb_str, y_pred_nb, labels=unique_labels)
    plot_confusion_matrix(cm_nb, unique_labels,
                          "Naive Bayes -- Karisiklik Matrisi (Confusion Matrix)",
                          "report/confusion_matrix_nb.png")

    # CRF Metrik Grafigi
    plot_metrics(crf_report_dict, "CRF", "report/metrics_crf.png")

    # NB Metrik Grafigi
    plot_metrics(nb_report_dict, "Naive Bayes", "report/metrics_nb.png")

    # Model Karşılaştırması
    plot_model_comparison(results, "report/model_comparison.png")

    # ===========================
    # 5. TAHMINLERI KAYDET
    # ===========================
    print("\n[5/5] Tahminler CoNLL formatinda kaydediliyor...")

    save_predictions_conll(test_sents, y_pred_crf, "report/predictions_crf.conll")

    # NB tahminlerini cumle bazli yeniden olustur
    nb_preds_by_sent = []
    idx = 0
    for words, _, _ in test_sents:
        end = idx + len(words)
        nb_preds_by_sent.append(list(y_pred_nb[idx:end]))
        idx = end
    save_predictions_conll(test_sents, nb_preds_by_sent, "report/predictions_nb.conll")

    # ===========================
    # ÖZET
    # ===========================
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    print(f"{'Model':<10} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 45)
    for model, res in results.items():
        print(f"{model:<10} {res['accuracy']:>10.4f} {res['macro_f1']:>10.4f} {res['weighted_f1']:>12.4f}")
    print("=" * 60)
    print("\nCikti dosyalari (report/ klasoru):")
    print("  confusion_matrix_crf.png  - CRF karisiklik matrisi")
    print("  confusion_matrix_nb.png   - Naive Bayes karisiklik matrisi")
    print("  metrics_crf.png           - CRF sinif bazli metrikler")
    print("  metrics_nb.png            - Naive Bayes sinif bazli metrikler")
    print("  model_comparison.png      - Karsilastirma grafigi")
    print("  predictions_crf.conll     - CRF tahminleri (CoNLL formati)")
    print("  predictions_nb.conll      - Naive Bayes tahminleri (CoNLL formati)")

if __name__ == "__main__":
    train_and_evaluate()
