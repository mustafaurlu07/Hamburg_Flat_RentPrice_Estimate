import pandas as pd
import numpy as np
import os
import json
import joblib
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_data():
    """Özellik mühendisliği yapılmış veriyi yükle"""
    df = pd.read_csv(
        r"C:\Users\musta\MLOPS\ML_RentEstimate\ModelGelistirmeveOptimizasyon\data\processed\hamburgrentflat_features.csv"
    )
    return df

def prepare_features(df):
    """Modelleme için özellikleri hazırla (kategorik değişkenler one-hot encode edilecek)"""
    y = df['cold_price']

    # Kullanılacak sütunlar
    feature_columns = [
        'city', 'district', 'object_age', 'flat_area', 'room_count',
        'distance_to_centre', 'price_per_sqm', 'age_category', 'distance_category'
    ]

    # Mevcut sütunları filtrele
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features].copy()

    # Kategorik sütunları tespit et
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-hot encode (drop_first=True ile multicollinearity azaltılır)
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Eksik değerleri doldur (numerik)
    X = X.fillna(X.mean())

    print(f"🔧 Hazırlanan özellikler: {X.columns.tolist()}")
    return X, y, X.columns.tolist()

# ------------------- OPTIMIZATION FUNCTIONS -------------------

def optimize_random_forest(X, y, cv_folds=5):
    print("🌲 Random Forest detaylı optimizasyonu...")

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 1.0],
        "bootstrap": [True, False],
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_grid, n_iter=50, cv=cv_folds,
        scoring="r2", n_jobs=-1, verbose=1, random_state=42
    )

    start_time = time.time()
    random_search.fit(X, y)
    elapsed = time.time() - start_time

    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_, elapsed

def optimize_gradient_boosting(X, y, cv_folds=5):
    print("🚀 Gradient Boosting detaylı optimizasyonu...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 0.9, 1.0],
        "max_features": ["sqrt", "log2", None],
    }

    gb = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        gb, param_distributions=param_grid, n_iter=50, cv=cv_folds,
        scoring="r2", n_jobs=-1, verbose=1, random_state=42
    )

    start_time = time.time()
    random_search.fit(X, y)
    elapsed = time.time() - start_time

    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_, elapsed

def optimize_knn(X, y, cv_folds=5):
    print("🤝 KNN Regressor detaylı optimizasyonu...")

    param_grid = {
        "n_neighbors": [3, 5, 7, 10, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2],  # Manhattan vs Euclidean
    }

    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(
        knn, param_grid, cv=cv_folds, scoring="r2", n_jobs=-1, verbose=1
    )

    start_time = time.time()
    grid_search.fit(X, y)
    elapsed = time.time() - start_time

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, elapsed

def optimize_decision_tree(X, y, cv_folds=5):
    print("🌳 Decision Tree detaylı optimizasyonu...")

    param_grid = {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 10],
        "max_features": ["sqrt", "log2", None],
    }

    dt = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(
        dt, param_grid, cv=cv_folds, scoring="r2", n_jobs=-1, verbose=1
    )

    start_time = time.time()
    grid_search.fit(X, y)
    elapsed = time.time() - start_time

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_, elapsed

# ------------------- FINAL EVALUATION -------------------

def final_evaluation(model, X, y, model_name, cv_folds=10):
    print(f"\n🎯 {model_name} final değerlendirmesi ({cv_folds} fold CV)...")

    cv_r2 = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv_folds, scoring="neg_mean_squared_error"))
    cv_mae = -cross_val_score(model, X, y, cv=cv_folds, scoring="neg_mean_absolute_error")

    results = {
        "r2_mean": float(cv_r2.mean()), "r2_std": float(cv_r2.std()),
        "rmse_mean": float(cv_rmse.mean()), "rmse_std": float(cv_rmse.std()),
        "mae_mean": float(cv_mae.mean()), "mae_std": float(cv_mae.std())
    }

    print("📊 Final Sonuçlar:")
    print(f"   R²:   {cv_r2.mean():.4f} (±{cv_r2.std()*2:.4f})")
    print(f"   RMSE: {cv_rmse.mean():.2f} (±{cv_rmse.std()*2:.2f})")
    print(f"   MAE:  {cv_mae.mean():.2f} (±{cv_mae.std()*2:.2f})")

    return results

# ------------------- MAIN -------------------

def main():
    print("🎯 En İyi Model Detaylı Optimizasyonu")
    print("="*50)

    df = load_data()
    X, y, feature_names = prepare_features(df)

    print(f"📊 Veri boyutu: {X.shape}")
    print(f"🔧 Özellikler: {feature_names}")

    # Önce best_model.pkl'i yükle
    best_model_path = r"C:\Users\musta\MLOPS\ML_RentEstimate\ModelGelistirmeveOptimizasyon\src\models\best_model.pkl"
    if not os.path.exists(best_model_path):
        print("❌ best_model.pkl bulunamadı! Önce model_gelistirme.py çalıştırın.")
        return

    best_model = joblib.load(best_model_path)
    best_model_name = type(best_model.named_steps["model"]).__name__
    print(f"📦 En iyi model: {best_model_name}")

    cv_folds = 5
    if best_model_name == "RandomForestRegressor":
        optimized_model, best_params, best_score, elapsed = optimize_random_forest(X, y, cv_folds)
    elif best_model_name == "GradientBoostingRegressor":
        optimized_model, best_params, best_score, elapsed = optimize_gradient_boosting(X, y, cv_folds)
    elif best_model_name == "KNeighborsRegressor":
        optimized_model, best_params, best_score, elapsed = optimize_knn(X, y, cv_folds)
    elif best_model_name == "DecisionTreeRegressor":
        optimized_model, best_params, best_score, elapsed = optimize_decision_tree(X, y, cv_folds)
    elif best_model_name == "LinearRegression":
        print("📈 Linear Regression için detaylı optimizasyon yok (parametre yok).")
        return
    else:
        print(f"❌ {best_model_name} için detaylı optimizasyon tanımlı değil!")
        return

    print(f"\n🚀 Optimizasyon Sonuçları:")
    print(f"   En iyi R² (CV): {best_score:.4f}")
    print(f"   Süre: {elapsed:.2f} sn")
    print(f"   Parametreler: {best_params}")

    # Final değerlendirme
    final_results = final_evaluation(optimized_model, X, y, best_model_name, cv_folds=10)

    # Kaydetme
    os.makedirs("models/detailed_optimized", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    joblib.dump(optimized_model, f"models/detailed_optimized/{best_model_name}_detailed.pkl")

    detailed_results = {
        "model_name": best_model_name,
        "best_params": best_params,
        "best_score_cv": best_score,
        "optimization_time_sec": elapsed,
        "final_evaluation": final_results,
        "features_used": feature_names,
    }

    with open("results/detailed_optimization_report.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    print("\n✅ Detaylı optimizasyon tamamlandı!")
    print(f"📁 Model kaydedildi: models/detailed_optimized/{best_model_name}_detailed.pkl")
    print(f"📊 Rapor: results/detailed_optimization_report.json")

if __name__ == "__main__":
    main()
