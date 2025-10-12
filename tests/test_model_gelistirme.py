import pandas as pd
from src import model_gelistirme

def test_load_data():
    df = model_gelistirme.load_data()
    assert not df.empty, "Dataframe is empty"
    assert "cold_price" in df.columns, "Target column missing"

def test_prepare_features():
    df = model_gelistirme.load_data()
    X, y, feature_names, preprocessor = model_gelistirme.prepare_features(df)
    
    # X ve y boyutları uyumlu olmalı
    assert len(X) == len(y), "X and y length mismatch"
    
    # Feature list boş olmamalı
    assert len(feature_names) > 0, "No features selected"
