import pandas as pd
import numpy as np
import os


def clean_data():
    """Hamburg 50 Km yaricapindaki kiralik daire veri setini temizle"""

    # Veriyi yükle
    df = pd.read_csv(r"data/raw/HamburgRentFlat50KmRadius.csv", encoding="cp1252")

    print(f"Orijinal veri boyutu: {df.shape}")
    print(f"Eksik degerler:\n{df.isnull().sum()}")

    # Eksik değerleri medyan ile doldur
    df["object_age"] = df["object_age"].fillna(df["object_age"].median())
    df["flat_area"] = df["flat_area"].fillna(df["flat_area"].median())
    df["room_count"] = df["room_count"].fillna(df["room_count"].median())
    df["distance_to_centre"] = df["distance_to_centre"].fillna(
        df["distance_to_centre"].median()
    )

    # city veya district boş olan satırları sil
    df_clean = df.dropna(subset=["city", "district"])

    print(f"\nTemizlenmis veri boyutu: {df_clean.shape}")
    print(f"Temizleme sonrası eksik degerler:\n{df_clean.isnull().sum()}")

    # Processed data dizinini oluştur
    os.makedirs("data/processed", exist_ok=True)

    # Temizlenmiş veriyi kaydet
    df_clean.to_csv("data/processed/hamburgrentflat_clean.csv", index=False)
    print(
        "Temizlenmis veri data/processed/hamburgrentflat_clean.csv dosyasına kaydedildi."
    )

    return df_clean


if __name__ == "__main__":
    clean_data()
