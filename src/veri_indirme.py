import pandas as pd
import os

def download_titanic_data():
    """Titanic veri setini yükle ve kaydet"""
    
    # Seaborn'dan Titanic veri setini yükle
    import seaborn as sns
    titanic_df = sns.load_dataset('titanic')
    
    # Raw data dizinini oluştur
    os.makedirs("data/raw", exist_ok=True)
    
    # Veriyi kaydet
    titanic_df.to_csv("data/raw/titanic_original.csv", index=False)
    
    print(f"Titanic veri seti indirildi: {titanic_df.shape[0]} satır, {titanic_df.shape[1]} sütun")
    print("Veri seti data/raw/titanic_original.csv dosyasına kaydedildi.")
    
    # Veri hakkında bilgi
    print("\nVeri seti özeti:")
    print(titanic_df.info())
    
    return titanic_df

if __name__ == "__main__":
    download_titanic_data()