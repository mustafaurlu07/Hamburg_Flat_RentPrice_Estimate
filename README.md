# Hamburg Flat Rent Prediction 🚀

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-yellow)]()

Bu proje, Hamburg 50 km yarıçapındaki kiralık daireler için kira tahmini yapan bir makine öğrenmesi modeli ve REST API servisidir. Swagger arayüzü ile kolay kullanım sağlar.

---

## 📂 Proje Yapısı
```
Hamburg_Flat_Rent_Estimate/
├── src/
│ ├── veri_temizleme.py # Veri temizleme modülü
│ ├── feature_engineering.py # Özellik mühendisliği
│ ├── model_gelistirme.py # Model geliştirme
│ ├── api.py # Flask API
│ └── predict.py # Tahmin sınıfı ve yardımcı fonksiyonlar
├── data/
│ ├── raw/ # Ham veri
│ └── processed/ # İşlenmiş veri ve özellikler
├── models/ # Eğitilmiş modeller
├── results/ # Sonuçlar ve metrikler
├── requirements.txt # Python bağımlılıkları
└── README.md # Bu dosya
```

---

## ⚙️ Özellikler

### Veri İşleme
- Ham veriyi yükleme ve temizleme
- Eksik değerleri median ile doldurma
- `city` ve `district` boş olan satırları silme
- Temizlenmiş veriyi kaydetme: `data/processed/hamburgrentflat_clean.csv`

### Özellik Mühendisliği
- `age_category` ve `distance_category` oluşturma
- Kategorik değişkenleri sayısal değerlere çevirme (`codes`)
- Gerekli feature set: 

['city', 'district', 'object_age', 'flat_area', 'room_count',
'distance_to_centre', 'age_category', 'distance_category']


### Model Geliştirme
- Kullanılan modeller:
- Random Forest
- Decision Tree
- Gradient Boosting
- KNN Regressor
- Linear Regression
- En iyi modelin otomatik seçimi ve kaydı (`models/best_model.pkl`)

### API
- Flask + Swagger ile REST API
- Endpoints:
- `GET /` → API bilgileri
- `POST /predict` → Kira tahmini
- `GET /sample` → Örnek istek formatı
- JSON input doğrulama ve hata yönetimi

---

## 🚀 Kurulum

1. Sanal ortam oluştur ve aktive et:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate


2. Paketleri yükle:
pip install -r requirements.txt

🏃 Kullanım
Model Eğitimi
python src/model_gelistirme.py

API Çalıştırma
python src/api.py
Swagger UI: http://localhost:5000/apidocs/

Örnek POST /predict isteği
curl -X POST "http://localhost:5000/predict" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "city": "Altona",
  "district": "Hamburg",
  "object_age": 50,
  "flat_area": 64,
  "room_count": 2,
  "distance_to_centre": 2
}'

Örnek API cevabı
{
  "status": "success",
  "input": {
    "city": "Altona",
    "district": "Hamburg",
    "object_age": 50,
    "flat_area": 64,
    "room_count": 2,
    "distance_to_centre": 2
  },
  "prediction": 1250.0,
  "currency": "EUR",
  "timestamp": "2025-10-20T20:00:00"
}

📦 Çıktılar

models/best_model.pkl → En iyi model

data/processed/ → Temizlenmiş ve özellik mühendisliği yapılmış veri

results/ → Model sonuçları ve metrikler (opsiyonel)

⚡ Lisans

MIT Lisansı altında dağıtılmıştır.