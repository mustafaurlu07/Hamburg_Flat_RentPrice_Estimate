import pandas as pd
import numpy as np
import os


def create_features():
    """Özellik mühendisliği yap"""

    # Temizlenmiş veriyi yükle
    df = pd.read_csv("data/processed/hamburgrentflat_clean.csv")

    print(f"Özellik mühendisliği öncesi boyut: {df.shape}")

    # Yeni özellikler oluştur

    # 1. Metrekare Fiyatı
    df['price_per_sqm'] = df['cold_price'] / df['flat_area']

    # 2. Yaş Kategorisi
    bins = [0, 10, 30, 60, 1000]
    labels = ['new', 'recent', 'middle_age', 'old']
    df['age_category'] = pd.cut(df['object_age'], bins=bins, labels=labels)

    # 3. Mesafe grupları
    df['distance_category'] = pd.cut(
        df['distance_to_centre'],
        bins=[0, 1, 3, 7, 100],
        labels=['very_close', 'close', 'medium', 'far']
    )

    """
    # 4. Şehir ve Bölge Adı
    df['city_district'] = df['city'] + '_' + df['district']
    """

    # Kategorik değişkenleri sayısala çevir
    categorical_columns = ['age_category', 'distance_category']

    for col in categorical_columns:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    print(f"Özellik mühendisliği sonrası boyut: {df.shape}")
    print("Yeni özellikler: price_per_sqm, age_category, distance_category")

    # Özellik mühendisliği yapılmış veriyi kaydet
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/hamburgrentflat_features.csv", index=False)

    print("Özellik mühendisliği tamamlandı!")
    print("Veri data/processed/hamburgrentflat_features.csv dosyasına kaydedildi.")

    # İlk 20 satırı göster
    print(df.head(20))

    return df


if __name__ == "__main__":
    create_features()