import pandas as pd
import pytest
from unittest.mock import patch
from src import veri_temizleme, feature_engineering

# Örnek veri
example_data = pd.DataFrame({
    "city": ["Norderstedt", "Kaltenkirchen"],
    "district": ["Harksheide", "Kaltenkirchen"],
    "cold_price": [1250, 1700],
    "object_age": [0, 23],
    "flat_area": [88, 171.31],
    "room_count": [2, 4],
    "distance_to_centre": [6.016562989, 0.411129512]
})

# ------------------ veri_temizleme test ------------------
def test_clean_data():
    # pd.read_csv fonksiyonunu örnek dataframe ile patchle
    with patch("src.veri_temizleme.pd.read_csv", return_value=example_data):
        df_clean = veri_temizleme.clean_data()
        
        # Temizlenmiş veri boş olmamalı
        assert not df_clean.empty, "Cleaned dataframe is empty"
        
        # Eksik değer kalmamalı
        assert df_clean.isnull().sum().sum() == 0, "There are still missing values"

# ------------------ feature_engineering test ------------------
def test_create_features():
    # Önce temizlenmiş veriyi yükle
    with patch("src.feature_engineering.pd.read_csv", return_value=example_data):
        df_features = feature_engineering.create_features()
        
        # Yeni özellikler oluşmuş mu?
        for col in ["price_per_sqm", "age_category", "distance_category"]:
            assert col in df_features.columns, f"{col} not in features"
        
        # Özellikler boş olmamalı
        assert not df_features.empty, "Feature-engineered dataframe is empty"
