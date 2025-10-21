#!/usr/bin/env python3
"""
Kira Tahmin Fonksiyonları - DEBUG aktif
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


class RentPredictor:
    """Kira tahmin sınıfı"""

    def __init__(self, model_path='models/best_model.pkl'):
        self.model = None
        self.model_info = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Modeli yükler"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")

            self.model = joblib.load(self.model_path)
            print(f"✅ Model yüklendi: {self.model_path}")

        except Exception as e:
            print(f"❌ Model yüklenirken hata: {e}")
            raise e

    def validate_input(self, data):
        """Input verilerini doğrular"""
        required_fields = [
            'city', 'district', 'object_age', 'flat_area',
            'room_count', 'distance_to_centre'
        ]

        print("🔍 Input doğrulama başlatıldı:", data)

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Gerekli alan eksik: {field}")

        return True

    def preprocess_input(self, data):
        """Gelen veriyi modele uygun hale getirir"""
        print("\n🟢 Gelen ham veri:", data)

        # Age category
        if data["object_age"] < 10:
            age_category = "new"
        elif data["object_age"] < 30:
            age_category = "recent"
        elif data["object_age"] < 60:
            age_category = "middle_age"
        else:
            age_category = "old"

        # Distance category
        if data["distance_to_centre"] < 1:
            distance_category = "very_close"
        elif data["distance_to_centre"] < 3:
            distance_category = "close"
        elif data["distance_to_centre"] < 7:
            distance_category = "medium"
        else:
            distance_category = "far"

        df_features = pd.DataFrame([{
            "city": data["city"],
            "district": data["district"],
            "object_age": float(data["object_age"]),
            "flat_area": float(data["flat_area"]),
            "room_count": int(data["room_count"]),
            "distance_to_centre": float(data["distance_to_centre"]),
            "age_category": age_category,
            "distance_category": distance_category
        }])

            # ✅ String kategorileri sayıya çeviriyoruz (model bunu bekliyor!)
        categorical_columns = ["age_category", "distance_category"]
        for col in categorical_columns:
            df_features[col] = pd.Categorical(df_features[col]).codes

        print("🛠️ Preprocessed DataFrame:\n", df_features)
        print("📊 Veri tipleri:\n", df_features.dtypes)

        return df_features

    def predict(self, data):
        """Kira tahmini yapar"""
        try:
            if self.model is None:
                raise RuntimeError("Model yüklenmemiş")

            # Validate
            self.validate_input(data)

            # Transform
            features = self.preprocess_input(data)

            print(f"📦 Model input shape: {features.shape}")
            print(f"📦 Model input preview:\n{features.head()}")

            # Predict
            prediction = self.model.predict(features)[0]
            print(f"✅ Tahmin sonucu: {prediction}")

            return {
                "prediction": round(float(prediction), 2),
                "currency": "EUR",
                "status": "success"
            }
        except Exception as e:
            print(f"❌ Tahmin sırasında hata: {e}")
            raise e


def create_sample_data():
    return {
        "rent_examples": [
            {
                "name": "Test Dairesi",
                "data": {
                    "city": "Hamburg",
                    "district": "Eimsbüttel",
                    "object_age": 5,
                    "flat_area": 60,
                    "room_count": 2,
                    "distance_to_centre": 2.5
                }
            }
        ]
    }


def test_predictor():
    predictor = RentPredictor()
    samples = create_sample_data()

    print("\n=== Test Başlıyor ===")
    for sample in samples["rent_examples"]:
        result = predictor.predict(sample["data"])
        print(f"{sample['name']} → Tahmin: €{result['prediction']}")


if __name__ == "__main__":
    test_predictor()
