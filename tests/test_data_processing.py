import os
import pandas as pd
import pytest
from src import veri_temizleme, feature_engineering

# ------------------ veri_temizleme test ------------------
def test_clean_data():
    df_clean = veri_temizleme.clean_data()
    
    # Temizlenmiş veri boş olmamalı
    assert not df_clean.empty, "Cleaned dataframe is empty"
    
    # Eksik değer kalmamalı
    assert df_clean.isnull().sum().sum() == 0, "There are still missing values"

# ------------------ feature_engineering test ------------------
def test_create_features():
    # Önce temizlenmiş veriyi yükle
    df_features = feature_engineering.create_features()
    
    # Yeni özellikler oluşmuş mu?
    for col in ["price_per_sqm", "age_category", "distance_category"]:
        assert col in df_features.columns, f"{col} not in features"

    # Özellikler boş olmamalı
    assert not df_features.empty, "Feature-engineered dataframe is empty"
