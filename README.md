# 🧠 Türkçe Metin Parçalama (Chunking) Projesi

> **Bursa Teknik Üniversitesi — Doğal Dil İşleme Dersi**  
> 2025–2026 Bahar Dönemi Projesi

---

## 📌 Proje Hakkında

Bu proje, Türkçe cümlelerdeki **isim öbeği (NP)**, **fiil öbeği (VP)**, **zarf öbeği (ADVP)** ve **edat öbeği (PP)** gibi yapısal birimleri otomatik olarak tanımlamayı amaçlayan bir **Chunk (Sözdizimsel Parçalama)** sistemidir.

Proje kapsamında iki farklı makine öğrenmesi modeli eğitilmiş ve karşılaştırılmıştır:

| Model | Açıklama |
|-------|----------|
| **CRF** (Conditional Random Field) | Dizilim etiketleme için özel tasarlanmış, bağlamı kullanan model |
| **Naive Bayes** | Olasılıksal, hızlı ve yorumlanabilir temel model |

Veri kaynağı olarak **Universal Dependencies (UD) Türkçe BOUN Treebank** kullanılmıştır.

---

## 📂 Proje Yapısı

```
Dogal_Dil_Isleme_Proje/
│
├── prepare_data.py        # UD Treebank'i indirir ve CoNLL formatına dönüştürür
├── chunker.py             # Modelleri eğitir, değerlendirir ve görselleştirir
├── requirements.txt       # Gerekli Python kütüphaneleri
├── Proje Raporu.pdf       # Proje teknik raporu
├── rapor.tex              # Raporun LaTeX kaynak kodu
│
├── data/                  # Veri dosyaları (otomatik oluşturulur)
│   ├── ud_turkish_train.conllu   # UD ham eğitim verisi
│   ├── ud_turkish_dev.conllu     # UD ham geliştirme verisi
│   ├── ud_turkish_test.conllu    # UD ham test verisi
│   ├── train.conll               # İşlenmiş eğitim verisi (BIO formatı)
│   └── test.conll                # İşlenmiş test verisi (BIO formatı)
│
└── report/                # Çıktı görselleri ve tahminler (otomatik oluşturulur)
    ├── confusion_matrix_crf.png   # CRF karmaşıklık matrisi
    ├── confusion_matrix_nb.png    # Naive Bayes karmaşıklık matrisi
    ├── metrics_crf.png            # CRF sınıf bazlı metrik grafiği
    ├── metrics_nb.png             # Naive Bayes sınıf bazlı metrik grafiği
    ├── model_comparison.png       # CRF vs Naive Bayes karşılaştırması
    ├── predictions_crf.conll      # CRF tahmin çıktısı
    └── predictions_nb.conll       # Naive Bayes tahmin çıktısı
```

---

## ⚙️ Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- pip

### Adım 1 — Sanal ortam oluştur (önerilir)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### Adım 2 — Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

`requirements.txt` içeriği:

```
sklearn-crfsuite>=0.3.6
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
```

---

## 🚀 Kullanım

Projeyi çalıştırmak için **iki adım** vardır:

### Adım 1 — Veriyi Hazırla

```bash
python prepare_data.py
```

Bu komut:
1. UD Türkçe BOUN Treebank verisini GitHub'dan otomatik indirir
2. CoNLL-U formatını BIO etiketli 3 sütunlu CoNLL formatına dönüştürür
3. `data/train.conll` ve `data/test.conll` dosyalarını oluşturur

> ⚠️ İlk çalıştırmada internet bağlantısı gereklidir (~10 MB veri indirilir).  
> Veri zaten mevcutsa tekrar indirilmez.

### Adım 2 — Modelleri Eğit ve Değerlendir

```bash
python chunker.py
```

Bu komut:
1. CoNLL verisini okur ve öznitelikleri çıkarır
2. CRF modelini eğitir (L-BFGS algoritması, c1=0.05, c2=0.05)
3. Naive Bayes modelini eğitir
4. Her iki modelin performansını raporlar
5. `report/` klasörüne grafikleri ve tahminleri kaydeder

---

## 📊 Etiket Sistemi (BIO Formatı)

Proje, **BIO (Beginning-Inside-Outside)** etiketleme şemasını kullanır:

| Etiket | Açıklama | Örnek |
|--------|----------|-------|
| `B-NP` | İsim öbeği başlangıcı | *"büyük"* (büyük şehir) |
| `I-NP` | İsim öbeği devamı | *"şehir"* (büyük şehir) |
| `B-VP` | Fiil öbeği başlangıcı | *"gidiyor"* |
| `I-VP` | Fiil öbeği devamı | |
| `B-ADVP` | Zarf öbeği başlangıcı | *"hızlıca"* |
| `I-ADVP` | Zarf öbeği devamı | |
| `B-PP` | Edat öbeği başlangıcı | *"için"* |
| `I-PP` | Edat öbeği devamı | |
| `O` | Herhangi bir öbeğe ait değil | |

---

## 🔧 Öznitelik Mühendisliği (Feature Engineering)

CRF modeli için her kelimeden şu öznitelikler çıkarılır:

- **Kelime özellikleri:** Kelimenin kendisi, küçük harf formu, büyük harf mi, baş harf büyük mü, rakam mı, uzunluğu
- **Son ekler (suffix):** Son 1, 2, 3, 4 karakter — *Türkçe'nin eklemeli yapısı için kritik!*
- **Ön ekler (prefix):** İlk 2, 3 karakter
- **Bağlam penceresi:** Önceki 2 ve sonraki 2 kelime ve özellikleri
- **Cümle pozisyonu:** Cümle başı (BOS) ve sonu (EOS) işaretleri

---

## 📈 Örnek Çıktı

Program çalıştıktan sonra terminalde şuna benzer bir özet görürsünüz:

```
============================================================
OZET
============================================================
Model      Accuracy   Macro F1  Weighted F1
---------------------------------------------
CRF          0.8912     0.7634       0.8875
Naive Bayes  0.7103     0.5421       0.7089
============================================================
```

---

## 🖼️ Çıktı Görselleri

Eğitim tamamlandıktan sonra `report/` klasöründe oluşan grafikler:

| Dosya | İçerik |
|-------|--------|
| `confusion_matrix_crf.png` | CRF modelinin sınıf karmaşıklık matrisi (ısı haritası) |
| `confusion_matrix_nb.png` | Naive Bayes modelinin karmaşıklık matrisi |
| `metrics_crf.png` | CRF — sınıf bazlı Precision / Recall / F1 bar grafikleri |
| `metrics_nb.png` | Naive Bayes — sınıf bazlı metrik grafikler |
| `model_comparison.png` | İki modelin yan yana karşılaştırması |

---

## 📚 Kullanılan Veri Kümesi

**Universal Dependencies (UD) Turkish BOUN Treebank**

- Kaynak: [github.com/UniversalDependencies/UD_Turkish-BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN)
- Lisans: CC BY-SA 4.0
- Format: CoNLL-U (9 sütun)
- İçerik: Boğaziçi Üniversitesi'nin derlediği Türkçe cümleler, morfolojik analiz ve bağımlılık ağaçları

---

## 🛠️ Kullanılan Teknolojiler

| Kütüphane | Amaç |
|-----------|------|
| `sklearn-crfsuite` | CRF modeli eğitimi ve tahmini |
| `scikit-learn` | Naive Bayes, metrikler, ölçekleme |
| `numpy` | Sayısal işlemler |
| `matplotlib` | Grafik çizimi |
| `seaborn` | Isı haritası (confusion matrix) |

---

## 🧪 Teknik Detaylar

### CRF Parametreleri
```python
CRF(
    algorithm="lbfgs",          # L-BFGS optimizasyon algoritması
    c1=0.05,                    # L1 düzenlileştirme
    c2=0.05,                    # L2 düzenlileştirme
    max_iterations=150,
    all_possible_transitions=True
)
```

### Naive Bayes Parametreleri
```python
MultinomialNB(alpha=0.1)        # Laplace düzleştirme
```
> Not: Naive Bayes negatif değer kabul etmediğinden `MinMaxScaler` ile öznitelikler [0,1] aralığına çekilir.

---

## 📝 Notlar ve Bilinen Kısıtlamalar

- Proje yalnızca **Türkçe** metin üzerinde test edilmiştir.
- Türkçe'nin eklemeli yapısı nedeniyle CRF modelinde suffix özellikleri özellikle önem taşımaktadır.
- Naive Bayes modeli, CRF'ye kıyasla daha düşük performans göstermektedir; bu beklenen bir sonuçtur çünkü NB token bağımsızlığını varsayar.
- `prepare_data.py` ilk çalıştırıldığında internet bağlantısı gereklidir.

---

## 👨‍🎓 Proje Bilgileri

- **Üniversite:** Bursa Teknik Üniversitesi
- **Ders:** Doğal Dil İşleme
- **Dönem:** 2025–2026 Bahar
- **Konu:** Türkçe Metin Parçalama (Chunking) — CRF ve Naive Bayes Karşılaştırması

---

*Bu proje, Doğal Dil İşleme dersi kapsamında akademik amaçlarla geliştirilmiştir.*
