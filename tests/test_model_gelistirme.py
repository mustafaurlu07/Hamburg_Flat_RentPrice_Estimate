import pandas as pd
from unittest.mock import patch
from src import model_gelistirme

# Örnek veri
example_features = pd.DataFrame({
    "city": ["Norderstedt", "Kaltenkirchen"],
    "district": ["Harksheide", "Kaltenkirchen"],
    "cold_price": [1250.0, 1700.0],
    "object_age": [0.0, 23.0],
    "flat_area": [88.0, 171.31],
    "room_count": [2.0, 4.0],
    "distance_to_centre": [6.016562989, 0.411129512],
    "price_per_sqm": [14.204545454545455, 9.923530441888973],
    "age_category": [-1, 1],
    "distance_category": [2, 0]
})

def test_load_data():
    # pd.read_csv veya load_data fonksiyonunu patchleyelim
    with patch("src.model_gelistirme.pd.read_csv", return_value=example_features):
        df = model_gelistirme.load_data()
        assert not df.empty, "Dataframe is empty"
        assert "cold_price" in df.columns, "Target column missing"

def test_prepare_features():
    with patch("src.model_gelistirme.pd.read_csv", return_value=example_features):
        df = model_gelistirme.load_data()
        X, y, feature_names, preprocessor = model_gelistirme.prepare_features(df)
        
        # X ve y boyutları uyumlu olmalı
        assert len(X) == len(y), "X and y length mismatch"
        
        # Feature list boş olmamalı
        assert len(feature_names) > 0, "No features selected"
