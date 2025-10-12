import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import os


def load_data():
    """Özellik mühendisliği yapılmış veriyi yükle"""
    input_path = "data/processed/hamburgrentflat_features.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Veri başarıyla yüklendi: {df.shape}")
    return df


def prepare_features(df):
    """Modelleme için özellikleri hazırla"""
    y = df["cold_price"]

    feature_columns = [
        "city",
        "district",
        "object_age",
        "flat_area",
        "room_count",
        "distance_to_centre",
        "price_per_sqm",
        "age_category",
        "distance_category",
    ]

    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_cols,
            ),
            ("num", StandardScaler(), numerical_cols),
        ]
    )

    print(f"Kullanılan özellikler: {available_features}")
    return X, y, available_features, preprocessor


def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    """Farklı regresyon modellerini eğit ve değerlendir"""
    models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=2,
            subsample=0.8,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"{name} modeli eğitiliyor...")

        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)

        results.append(
            {
                "Model": name,
                "Eğitim R²": round(r2_train, 3),
                "Test R²": round(r2_test, 3),
                "Fark": round(r2_train - r2_test, 3),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
            }
        )

        trained_models[name] = pipeline

    results_df = pd.DataFrame(results)
    return trained_models, results_df


def save_models_and_results(trained_models, results_df, feature_names):
    """Modelleri ve sonuçları kaydet"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name, model in trained_models.items():
        joblib.dump(model, f"models/{name}.pkl")

    results_df.to_csv("results/model_results.csv", index=False)

    with open("results/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # En iyi modeli Test R²'ye göre seç
    best_model_name = results_df.sort_values(by="Test R²", ascending=False).iloc[0][
        "Model"
    ]
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, "models/best_model.pkl")

    print(f"\n✅ En iyi model: {best_model_name}")
    print(results_df)


def main():
    print("🚀 Model geliştirme başlatılıyor...")
    df = load_data()

    X, y, feature_names, preprocessor = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained_models, results_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    save_models_and_results(trained_models, results_df, feature_names)
    print("\n🏁 Model geliştirme tamamlandı!")


if __name__ == "__main__":
    main()
