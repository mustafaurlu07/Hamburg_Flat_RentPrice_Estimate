import pandas as pd
import numpy as np
import os

def clean_data():
    """Titanic veri setini temizle"""
    
    # Veriyi yükle
    df = pd.read_csv("data/raw/titanic_original.csv")
    
    print(f"Orijinal veri boyutu: {df.shape}")
    print(f"Eksik değerler:\n{df.isnull().sum()}")
    
    # Gereksiz sütunları kaldır
    columns_to_drop = ['deck', 'embark_town', 'alive', 'alone']
    df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Yaş eksik değerlerini medyan ile doldur
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    
    # Embarked eksik değerlerini mod ile doldur
    if 'embarked' in df_clean.columns:
        df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])
    
    # Fare eksik değerlerini medyan ile doldur
    df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    # Adult_male sütununu bool'dan int'e çevir
    if 'adult_male' in df_clean.columns:
        df_clean['adult_male'] = df_clean['adult_male'].astype(int)
    
    print(f"\nTemizlenmiş veri boyutu: {df_clean.shape}")
    print(f"Temizleme sonrası eksik değerler:\n{df_clean.isnull().sum()}")
    
    # Processed data dizinini oluştur
    os.makedirs("data/processed", exist_ok=True)
    
    # Temizlenmiş veriyi kaydet
    df_clean.to_csv("data/processed/titanic_clean.csv", index=False)
    
    print("Temizlenmiş veri data/processed/titanic_clean.csv dosyasına kaydedildi.")
    
    return df_clean

if __name__ == "__main__":
    clean_data()