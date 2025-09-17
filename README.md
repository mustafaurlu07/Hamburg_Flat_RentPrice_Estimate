# Titanic Survival Prediction - Model Development & Optimization

Bu proje, Titanic veri seti kullanarak hayatta kalma tahmini yapan makine öğrenmesi modelleri geliştirmeyi ve optimize etmeyi amaçlamaktadır.

## Proje Yapısı

```
ornek_4/
├── src/
│   ├── veri_indirme.py          # Veri indirme modülü
│   ├── veri_temizleme.py        # Veri temizleme modülü
│   ├── feature_engineering.py   # Özellik mühendisliği
│   ├── model_gelistirme.py      # Model geliştirme
│   └── model_optimizasyon.py    # Model optimizasyonu
├── data/
│   ├── raw/                     # Ham veri
│   └── processed/               # İşlenmiş veri
├── models/                      # Eğitilmiş modeller
├── results/                     # Sonuçlar ve metrikler
├── dvc.yaml                     # DVC pipeline tanımı
├── requirements.txt             # Python bağımlılıkları
└── README.md                    # Bu dosya
```

## Özellikler

### Veri İşleme
- Titanic veri setinin otomatik indirilmesi
- Eksik değerlerin akıllı doldurulması
- Veri temizleme ve ön işleme

### Özellik Mühendisliği
- Aile büyüklüğü hesaplama
- Yaş ve ücret grupları oluşturma
- Unvan çıkarma
- Kategorik değişkenlerin kodlanması

### Model Geliştirme
Aşağıdaki modeller eğitilir ve karşılaştırılır:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- Gradient Boosting
- Naive Bayes

### Akıllı Model Optimizasyonu
- En iyi performans gösteren modeli otomatik belirleme
- İki aşamalı optimizasyon (RandomSearch + GridSearch)
- Detaylı cross-validation (10-fold)
- Kapsamlı metrik değerlendirmesi (Accuracy, ROC-AUC, Precision, Recall, F1)

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. DVC'yi başlatın (eğer henüz başlatılmamışsa):
```bash
dvc init
```

## Kullanım

### 🚀 Önerilen Yöntem: Otomatik Pipeline Runner
```bash
# Tüm pipeline'ı çalıştır
python run_pipeline.py

# DVC pipeline kullan
python run_pipeline.py --dvc

# Sadece belirli bir adımı çalıştır
python run_pipeline.py --step models
python run_pipeline.py --step optimize
```

### Geleneksel Yöntemler

#### DVC Pipeline
```bash
dvc repro
```

#### Aşama Aşama Manuel Çalıştırma

1. Veri indirme:
```bash
python src/veri_indirme.py
```

2. Veri temizleme:
```bash
python src/veri_temizleme.py
```

3. Özellik mühendisliği:
```bash
python src/feature_engineering.py
```

4. Model geliştirme:
```bash
python src/model_gelistirme.py
```

5. Model optimizasyonu:
```bash
python src/model_optimizasyon.py
```

## Çıktılar

### Modeller
- `models/`: Temel modeller
- `models/optimized/`: Optimize edilmiş modeller
- `models/best_model.pkl`: En iyi temel model
- `models/optimized/best_optimized_model.pkl`: En iyi optimize edilmiş model

### Sonuçlar
- `results/model_results.json`: Temel model performans sonuçları
- `results/optimization_results.json`: Optimizasyon sonuçları
- `results/best_model_info.json`: En iyi model bilgileri
- `results/best_optimized_model_info.json`: En iyi optimize edilmiş model bilgileri

## DVC Pipeline Aşamaları

1. **data_download**: Titanic veri setini indir
2. **data_clean**: Veriyi temizle ve ön işle
3. **feature_engineering**: Yeni özellikler oluştur
4. **model_development**: Farklı modelleri eğit ve değerlendir
5. **model_optimization**: Hiperparametre optimizasyonu yap

## Metrikler

- **Accuracy**: Doğru tahmin oranı
- **ROC-AUC**: Receiver Operating Characteristic - Area Under Curve
- **Cross-validation**: 5-fold stratified cross-validation
- **Confusion Matrix**: Karışıklık matrisi
- **Classification Report**: Detaylı sınıflandırma raporu

## Geliştirme

Yeni özellikler eklemek veya mevcut modelleri geliştirmek için:

1. Yeni bir script oluşturun `src/` dizininde
2. `dvc.yaml` dosyasına yeni aşama ekleyin
3. Pipeline'ı yeniden çalıştırın: `dvc repro`

## Lisans

Bu proje eğitim amaçlıdır ve MIT lisansı altındadır.