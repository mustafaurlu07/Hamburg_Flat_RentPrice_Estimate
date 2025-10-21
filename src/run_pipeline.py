import subprocess
import sys  # ✅ Bunu ekle

scripts = [
    "src/veri_temizleme.py",
    "src/feature_engineering.py",
    "src/model_gelistirme.py",
    "src/best_model_optimize.py",
]

for script in scripts:
    print(f"🔹 Çalıştırılıyor: {script}")
    subprocess.run([sys.executable, script], check=True)  # ✅ BURAYI DÜZELTTİK

print("✅ Tüm pipeline başarıyla tamamlandı!")
