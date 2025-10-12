import pandas as pd
import joblib

# Load processed data
df_validate = pd.read_csv(
    r"C:\Users\musta\MLOPS\ML_RentEstimate\ModelGelistirmeveOptimizasyon\data\processed\hamburgrentflat_features.csv",
    encoding="utf-8"
).head(20)

# Load trained model (pipeline already handles preprocessing)
best_model = joblib.load(r"models/best_model.pkl")

# Define feature columns exactly as during training
feature_columns = [
    'city', 'district', 'object_age', 'flat_area', 'room_count',
    'distance_to_centre', 'price_per_sqm', 'age_category', 'distance_category'
]
X = df_validate[feature_columns]

# Predict
df_validate["predicted_cold_price"] = best_model.predict(X)

# Show result
print(df_validate[["city", "district", "cold_price", "predicted_cold_price"]])
