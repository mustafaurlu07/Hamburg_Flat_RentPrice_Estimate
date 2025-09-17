import pandas as pd
import numpy as np
import os

def create_features():
    """Özellik mühendisliği yap"""
    
    # Temizlenmiş veriyi yükle
    df = pd.read_csv("data/processed/titanic_clean.csv")
    
    print(f"Özellik mühendisliği öncesi boyut: {df.shape}")
    
    # Yeni özellikler oluştur
    
    # 1. Aile büyüklüğü
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    
    # 2. Yalnız seyahat edip etmediği
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    # 3. Yaş grupları
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'Middle_age', 'Senior'])
    
    # 4. Fare grupları
    df['fare_group'] = pd.cut(df['fare'], 
                             bins=[0, 7.91, 14.45, 31, 1000], 
                             labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # 5. Unvan çıkarma (eğer name sütunu varsa)
    if 'name' in df.columns:
        df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Nadir unvanları 'Other' olarak grupla
        title_counts = df['title'].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        df.loc[df['title'].isin(rare_titles), 'title'] = 'Other'
    
    # Kategorik değişkenleri sayısala çevir
    categorical_columns = ['sex', 'embarked', 'age_group', 'fare_group']
    if 'title' in df.columns:
        categorical_columns.append('title')
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
    
    # Class sütununu pclass olarak yeniden adlandır (eğer varsa)
    if 'class' in df.columns and 'pclass' not in df.columns:
        df = df.rename(columns={'class': 'pclass'})
    
    print(f"Özellik mühendisliği sonrası boyut: {df.shape}")
    print(f"Yeni özellikler: family_size, is_alone, age_group, fare_group")
    if 'title' in df.columns:
        print("Title özelliği de eklendi.")
    
    # Özellik mühendisliği yapılmış veriyi kaydet
    df.to_csv("data/processed/titanic_features.csv", index=False)
    
    print("Özellik mühendisliği tamamlandı!")
    print("Veri data/processed/titanic_features.csv dosyasına kaydedildi.")
    
    return df

if __name__ == "__main__":
    create_features()